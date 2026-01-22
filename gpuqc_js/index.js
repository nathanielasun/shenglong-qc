/**
 * GPU Accelerated Quantum Circuit Simulator (JavaScript/TensorFlow.js)
 *
 * A high-performance statevector quantum circuit simulator supporting
 * CPU, CUDA, Metal, and WebGL backends via TensorFlow.js.
 *
 * Features:
 * - GPU acceleration via TensorFlow.js (CUDA, Metal, WebGL)
 * - Efficient strided gate application
 * - Mid-circuit measurements
 * - Parametric gates (Rx, Ry, Rz, U, P)
 * - Controlled gates (CNOT, CZ, multi-controlled)
 * - Circuit visualization (ASCII diagrams)
 *
 * Basic Usage:
 * ```javascript
 * import { Circuit } from 'gpuqc_js';
 *
 * const qc = new Circuit(2, { device: 'gpu' });
 * qc.h(0).cnot(0, 1);
 * await qc.execute(1000);
 * console.log(qc.getCounts());
 * ```
 *
 * @author Nathaniel Sun
 * @version 0.1.0
 */

// Core classes
export { Gate, Complex } from './gates.js';
export { Circuit, configureBackend } from './circuit.js';

// Visualization
export {
  ASCIICircuitDrawer,
  ProbabilityVisualizer,
  CircuitVisualizer,
  printCircuit,
  drawCircuit,
  circuitSummary,
} from './visualization.js';

// Utilities
export {
  // Memory monitoring
  getMemoryInfo,
  getMemoryUsageMB,
  printMemoryStats,
  estimateStatevectorMemory,

  // Device utilities
  getAvailableDevices,
  getCurrentBackend,
  getBackendInfo,

  // I/O
  saveStatevector,
  loadStatevector,
  saveStatevectorBinary,
  loadStatevectorBinary,

  // Analysis
  fidelity,
  traceDistance,
  getNonzeroAmplitudes,
  printStatevector,

  // Measurement utilities
  countsToProbabilities,
  getTopOutcomes,
  printCounts,
} from './utils.js';

// Examples
export { runExamples } from './examples.js';

// Version info
export const VERSION = '0.1.0';
export const AUTHOR = 'Nathaniel Sun';

// Default export
export default {
  VERSION,
  AUTHOR,
};
