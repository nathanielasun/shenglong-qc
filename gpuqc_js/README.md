# GPU Accelerated Quantum Circuit Simulator (JavaScript)

A high-performance statevector quantum circuit simulator with GPU acceleration support for CUDA, Metal, and WebGL via TensorFlow.js.

## Features

- **GPU Acceleration**: Supports NVIDIA CUDA, Apple Metal, and WebGL backends
- **TensorFlow.js Backend**: Leverages TensorFlow.js for cross-platform GPU support
- **Efficient Gate Application**: Strided indexing avoids large tensor products
- **Mid-Circuit Measurements**: Full support for measurement and collapse during execution
- **Comprehensive Gate Set**: Clifford gates, rotation gates, controlled gates, and custom unitaries
- **Fluent API**: Method chaining for clean circuit construction
- **Circuit Visualization**: ASCII diagrams for terminal display

## Installation

### Requirements

- Node.js 18+
- npm or yarn

### Install

```bash
npm install
```

### GPU Support

For native GPU acceleration (CUDA on Linux/Windows, Metal on macOS):

```bash
# For GPU support (CUDA/Metal)
npm install @tensorflow/tfjs-node-gpu

# For optimized CPU (recommended for development)
npm install @tensorflow/tfjs-node
```

### Package Structure

```
gpuqc_js/
├── package.json      # Package configuration
├── index.js          # Main exports
├── gates.js          # Gate class with quantum gate definitions
├── circuit.js        # Circuit class for building and executing circuits
├── visualization.js  # ASCII circuit diagrams
├── utils.js          # Utility functions
├── examples.js       # Example circuits
└── README.md         # Documentation
```

## Quick Start

```javascript
import { Circuit } from 'gpuqc_js';

// Create a 2-qubit Bell state
const qc = new Circuit(2, { device: 'gpu' });
qc.h(0).cnot(0, 1);

// Execute and measure
await qc.execute(1000);
console.log(qc.getCounts());  // { '00': ~500, '11': ~500 }

// Cleanup
qc.dispose();
```

### Running Examples

```bash
npm run examples
# or
node examples.js
```

## API Reference

### Circuit Class

The main class for building and executing quantum circuits.

#### Constructor

```javascript
new Circuit(size, options)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `number` | Number of qubits |
| `options.device` | `string` | `'gpu'`, `'cpu'`, `'webgl'`, or `'wasm'` |

#### Adding Gates

**General Method:**
```javascript
qc.add(gate, target, { control, controls, angle, angles })
```

**Shorthand Methods:**

| Method | Description | Example |
|--------|-------------|---------|
| `h(target)` | Hadamard gate | `qc.h(0)` |
| `x(target)` | Pauli-X (NOT) | `qc.x(0)` |
| `y(target)` | Pauli-Y | `qc.y(0)` |
| `z(target)` | Pauli-Z | `qc.z(0)` |
| `s(target)` | S gate (√Z) | `qc.s(0)` |
| `t(target)` | T gate (√S) | `qc.t(0)` |
| `rx(target, angle)` | X-rotation | `qc.rx(0, Math.PI/2)` |
| `ry(target, angle)` | Y-rotation | `qc.ry(0, Math.PI/4)` |
| `rz(target, angle)` | Z-rotation | `qc.rz(0, Math.PI)` |
| `p(target, angle)` | Phase gate | `qc.p(0, Math.PI/4)` |
| `u(target, θ, φ, λ)` | Universal U3 | `qc.u(0, Math.PI/2, 0, Math.PI)` |
| `cnot(ctrl, tgt)` | CNOT gate | `qc.cnot(0, 1)` |
| `cx(ctrl, tgt)` | CNOT (alias) | `qc.cx(0, 1)` |
| `cz(ctrl, tgt)` | Controlled-Z | `qc.cz(0, 1)` |
| `swap(q1, q2)` | SWAP gate | `qc.swap(0, 1)` |
| `measure(target)` | Mid-circuit measurement | `qc.measure(0)` |

#### Execution

```javascript
await qc.execute(shots, options)
```

| Parameter | Description |
|-----------|-------------|
| `shots` | Number of measurement samples |
| `options.cache` | Cache statevector before first MCM |

**Note:** `execute()` is async and returns a Promise.

#### Results

| Method | Returns | Description |
|--------|---------|-------------|
| `getCounts()` | `Object` | Measurement counts (little-endian) |
| `getCountsBigEndian()` | `Object` | Measurement counts (big-endian) |
| `getStatevector()` | `Object` | `{ real: [], imag: [] }` |
| `getProbabilities()` | `number[]` | Probability distribution |

#### Circuit Information

| Method | Returns | Description |
|--------|---------|-------------|
| `depth()` | `number` | Circuit depth |
| `gateCount()` | `Object` | Count of each gate type |
| `printCircuit()` | void | Print text representation |
| `length` | `number` | Total number of gates |
| `dispose()` | void | Free tensor resources |

### Gate Class

Low-level gate definitions and application methods.

```javascript
import { Gate, Complex } from 'gpuqc_js';

const gate = new Gate();

// Complex number utilities
const c = Complex.create(1, 0);    // 1 + 0i
const exp = Complex.exp(Math.PI/4); // e^(i*π/4)
```

#### Available Gates

**Fixed Gates:**
- `gate.H` - Hadamard
- `gate.X`, `gate.Y`, `gate.Z` - Pauli gates
- `gate.S`, `gate.Sdg` - S and S-dagger
- `gate.T`, `gate.Tdg` - T and T-dagger
- `gate.SX` - √X gate
- `gate.I` - Identity

**Parametric Gates (Methods):**
- `gate.Rx(angle)` - X-rotation
- `gate.Ry(angle)` - Y-rotation
- `gate.Rz(angle)` - Z-rotation
- `gate.P(phi)` - Phase shift
- `gate.Ph(phase)` - Global phase
- `gate.U(theta, phi, lam)` - Universal U3 gate

**Two-Qubit Gates:**
- `gate.CNOT(state, control, target)`
- `gate.CZ(state, control, target)`
- `gate.SWAP(state, qubit1, qubit2)`

### Visualization

```javascript
import { printCircuit, CircuitVisualizer, circuitSummary } from 'gpuqc_js';

// Print ASCII circuit
printCircuit(qc);

// Full visualizer
const vis = new CircuitVisualizer(qc);
vis.printCircuit();
vis.printProbabilities();
vis.printCounts();
vis.summary();

// Quick summary
circuitSummary(qc);
```

#### ASCII Output Example

```
q0: ──[H]──●─────────
           │
q1: ───────X──●──────
              │
q2: ──────────X──────
```

### Utility Functions

```javascript
import {
  getMemoryUsageMB,
  printMemoryStats,
  estimateStatevectorMemory,
  getAvailableDevices,
  getCurrentBackend,
  fidelity,
  traceDistance,
  printStatevector,
  countsToProbabilities,
} from 'gpuqc_js';

// Memory
console.log(`Memory: ${getMemoryUsageMB()} MB`);
printMemoryStats();

// Devices
const devices = await getAvailableDevices();
console.log(devices);

// Analysis
const f = fidelity(state1, state2);
printStatevector(qc.getStatevector());
```

## Examples

### Bell State

```javascript
import { Circuit } from 'gpuqc_js';

const qc = new Circuit(2, { device: 'gpu' });
qc.h(0).cnot(0, 1);
await qc.execute(1000);
console.log(qc.getCounts());
// Output: { '00': 512, '11': 488 }
```

### GHZ State

```javascript
const qc = new Circuit(3);
qc.h(0).cnot(0, 1).cnot(1, 2);
await qc.execute(1000);
console.log(qc.getCounts());
// Output: { '000': 495, '111': 505 }
```

### Rotation Gates

```javascript
const qc = new Circuit(1);
qc.ry(0, Math.PI / 4);
await qc.execute(1000);
// ~85% |0>, ~15% |1>
```

### Method Chaining

```javascript
const result = new Circuit(2, { device: 'gpu' })
  .h(0)
  .cnot(0, 1)
  .rz(0, Math.PI / 4);

await result.execute(1000);
console.log(result.getCounts());
```

## Performance Notes

### Memory Requirements

| Qubits | Amplitudes | Memory |
|--------|------------|--------|
| 10 | 1,024 | 8 KB |
| 15 | 32,768 | 256 KB |
| 20 | 1,048,576 | 8 MB |
| 25 | 33,554,432 | 256 MB |
| 30 | 1,073,741,824 | 8 GB |

### Backend Selection

- **tensorflow-gpu**: Best performance for CUDA (Linux/Windows) or Metal (macOS)
- **tensorflow**: Optimized CPU via native bindings
- **webgl**: Browser-based GPU acceleration
- **cpu**: Pure JavaScript fallback

```javascript
import { configureBackend } from 'gpuqc_js';

// Explicitly configure backend
await configureBackend('gpu');  // Auto-detect CUDA/Metal
await configureBackend('cpu');  // Use CPU
```

## Comparison with Python Version

This package (`gpuqc_js`) is a JavaScript port of the Python `mps_accel_qc` package. Key differences:

| Feature | Python (mps_accel_qc) | JavaScript (gpuqc_js) |
|---------|----------------------|----------------------|
| Backend | PyTorch | TensorFlow.js |
| GPU | CUDA, MPS | CUDA, Metal, WebGL |
| Async | Synchronous | Async (`await`) |
| Complex Numbers | Native | Custom implementation |
| Visualization | ASCII + Matplotlib | ASCII only |

## Qubit Indexing Convention

Uses **little-endian** bit ordering:
- Qubit 0 is the **least significant bit** (rightmost)
- The state `|abc>` means qubit 2 = a, qubit 1 = b, qubit 0 = c

```javascript
const qc = new Circuit(3);
qc.x(0);  // Sets qubit 0 to |1>
await qc.execute(100);
console.log(qc.getCounts());  // { '001': 100 }
```

## License

MIT License

## Author

Nathaniel Sun
