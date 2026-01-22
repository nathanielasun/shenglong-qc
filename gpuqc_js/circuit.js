/**
 * Quantum Circuit Construction and Execution.
 *
 * This module provides the Circuit class for building quantum circuits,
 * executing them via statevector simulation, and sampling measurement outcomes.
 * Supports CPU, CUDA, and Metal backends via TensorFlow.js.
 *
 * @author Nathaniel Sun
 */

import * as tf from '@tensorflow/tfjs';
import { Gate } from './gates.js';

/**
 * Configure TensorFlow.js backend based on user preference.
 * @param {string} device - Device type ('gpu', 'cpu', 'webgl', 'wasm')
 * @returns {Promise<string>} - The active backend name
 */
export async function configureBackend(device) {
  const deviceLower = (device || 'cpu').toLowerCase();

  try {
    if (deviceLower === 'gpu' || deviceLower === 'cuda' || deviceLower === 'metal') {
      // Try to use tfjs-node-gpu for native CUDA/Metal support
      try {
        await import('@tensorflow/tfjs-node-gpu');
        await tf.setBackend('tensorflow');
        console.log('Using TensorFlow.js with GPU (CUDA/Metal)');
        return 'tensorflow';
      } catch {
        // Fall back to WebGL
        await tf.setBackend('webgl');
        console.log('Using WebGL backend');
        return 'webgl';
      }
    } else if (deviceLower === 'webgl') {
      await tf.setBackend('webgl');
      console.log('Using WebGL backend');
      return 'webgl';
    } else if (deviceLower === 'wasm') {
      await tf.setBackend('wasm');
      console.log('Using WASM backend');
      return 'wasm';
    } else {
      // CPU fallback - try tfjs-node first for better performance
      try {
        await import('@tensorflow/tfjs-node');
        await tf.setBackend('tensorflow');
        console.log('Using TensorFlow.js with CPU (native)');
        return 'tensorflow';
      } catch {
        await tf.setBackend('cpu');
        console.log('Using CPU backend');
        return 'cpu';
      }
    }
  } catch (err) {
    console.warn(`Failed to set backend ${deviceLower}, using default:`, err.message);
    return tf.getBackend();
  }
}

/**
 * Gate name aliases for user convenience.
 */
const GATE_ALIASES = {
  'NOT': 'X',
  'PHASE': 'P',
  'RX': 'Rx',
  'RY': 'Ry',
  'RZ': 'Rz',
};

/**
 * A quantum circuit for statevector simulation.
 *
 * This class provides methods for constructing quantum circuits by adding
 * gates, executing the circuit to obtain the final statevector, and
 * sampling measurement outcomes.
 *
 * @example
 * const qc = new Circuit(2, { device: 'gpu' });
 * qc.h(0).cnot(0, 1);
 * await qc.execute(1000);
 * console.log(qc.getCounts());
 */
export class Circuit {
  /**
   * Create a new quantum circuit.
   *
   * @param {number} size - Number of qubits
   * @param {Object} options - Configuration options
   * @param {string} options.device - Device type ('gpu', 'cpu', 'webgl', 'wasm')
   */
  constructor(size, options = {}) {
    this.size = size;
    this.device = options.device || 'cpu';
    this.gateObj = new Gate();

    // Build gate registry
    this._gateRegistry = this._buildGateRegistry();

    // Circuit state
    this.state = null;
    this.measurements = null;
    this.circuit = [];
    this._backendConfigured = false;
  }

  /**
   * Build registry mapping gate names to implementations.
   * @private
   */
  _buildGateRegistry() {
    return {
      // Projection operators
      '0': this.gateObj.zero,
      '1': this.gateObj.one,

      // Single-qubit Clifford gates
      'H': this.gateObj.H,
      'I': this.gateObj.I,
      'X': this.gateObj.X,
      'Y': this.gateObj.Y,
      'Z': this.gateObj.Z,
      'S': this.gateObj.S,
      'Sdg': this.gateObj.Sdg,
      'SX': this.gateObj.SX,

      // T gates
      'T': this.gateObj.T,
      'Tdg': this.gateObj.Tdg,

      // Parametric gates (methods)
      'P': (angle) => this.gateObj.P(angle),
      'Ph': (angle) => this.gateObj.Ph(angle),
      'Rx': (angle) => this.gateObj.Rx(angle),
      'Ry': (angle) => this.gateObj.Ry(angle),
      'Rz': (angle) => this.gateObj.Rz(angle),
      'U': (theta, phi, lam) => this.gateObj.U(theta, phi, lam),

      // Two-qubit gates (handled specially)
      'CNOT': 'CNOT',
      'CX': 'CNOT',
      'CZ': 'CZ',
      'SWAP': 'SWAP',

      // Measurement
      'MCM': 'MCM',
      'MEASURE': 'MCM',
    };
  }

  /**
   * Normalize gate name using aliases.
   * @private
   */
  _normalizeGateName(name) {
    const upper = name.toUpperCase();
    return GATE_ALIASES[upper] || name;
  }

  // ===========================================================================
  // Circuit Construction
  // ===========================================================================

  /**
   * Add a gate to the circuit.
   *
   * @param {string} gate - Gate name (e.g., 'H', 'X', 'CNOT', 'Rx')
   * @param {number} target - Target qubit index
   * @param {Object} options - Additional options
   * @param {number} options.control - Single control qubit
   * @param {number[]} options.controls - Multiple control qubits
   * @param {number} options.angle - Single angle parameter
   * @param {number[]} options.angles - Multiple angle parameters
   * @returns {Circuit} this, for method chaining
   */
  add(gate, target, options = {}) {
    const gateName = this._normalizeGateName(gate);

    if (!(gateName in this._gateRegistry)) {
      throw new Error(`Unknown gate '${gate}'. Available: ${Object.keys(this._gateRegistry).join(', ')}`);
    }

    if (target < 0 || target >= this.size) {
      throw new Error(`Target qubit ${target} out of range for ${this.size}-qubit circuit`);
    }

    const meta = {
      name: gateName,
      target: target,
    };

    // Handle controls
    let controls = options.controls;
    if (options.control !== undefined) {
      if (controls !== undefined) {
        throw new Error("Specify either 'control' or 'controls', not both");
      }
      controls = [options.control];
    }

    if (controls) {
      for (const c of controls) {
        if (c < 0 || c >= this.size) {
          throw new Error(`Control qubit ${c} out of range`);
        }
        if (c === target) {
          throw new Error('Control and target qubits must be different');
        }
      }
      meta.controls = controls;
    }

    // Handle angles
    if (options.angle !== undefined) {
      if (options.angles !== undefined) {
        throw new Error("Specify either 'angle' or 'angles', not both");
      }
      meta.angles = options.angle;
    } else if (options.angles !== undefined) {
      meta.angles = options.angles;
    }

    this.circuit.push(meta);
    return this;
  }

  // Shorthand methods
  h(target) { return this.add('H', target); }
  x(target) { return this.add('X', target); }
  y(target) { return this.add('Y', target); }
  z(target) { return this.add('Z', target); }
  s(target) { return this.add('S', target); }
  t(target) { return this.add('T', target); }

  rx(target, angle) { return this.add('Rx', target, { angle }); }
  ry(target, angle) { return this.add('Ry', target, { angle }); }
  rz(target, angle) { return this.add('Rz', target, { angle }); }
  p(target, angle) { return this.add('P', target, { angle }); }

  u(target, theta, phi, lam) {
    return this.add('U', target, { angles: [theta, phi, lam] });
  }

  cx(control, target) { return this.add('CNOT', target, { control }); }
  cnot(control, target) { return this.cx(control, target); }
  cz(control, target) { return this.add('CZ', target, { control }); }
  swap(qubit1, qubit2) { return this.add('SWAP', qubit2, { control: qubit1 }); }

  measure(target) { return this.add('MCM', target); }

  barrier() {
    this.circuit.push({ name: 'BARRIER', target: -1 });
    return this;
  }

  // ===========================================================================
  // Circuit Execution
  // ===========================================================================

  /**
   * Execute the circuit and sample measurement outcomes.
   *
   * @param {number} shots - Number of measurement samples
   * @param {Object} options - Execution options
   * @param {boolean} options.cache - Cache statevector before first MCM
   * @returns {Promise<Circuit>} this, for method chaining
   */
  async execute(shots = 1024, options = {}) {
    // Configure backend if not done
    if (!this._backendConfigured) {
      await configureBackend(this.device);
      this._backendConfigured = true;
    }

    this.measurements = new Array(2 ** this.size).fill(0);

    const mcmIdx = this._findFirstMCM();

    if (mcmIdx !== null) {
      // Circuit has mid-circuit measurement
      if (options.cache && mcmIdx > 0) {
        this._runCircuit(null, 0, mcmIdx);
        const cachedReal = this.state.real.clone();
        const cachedImag = this.state.imag.clone();

        for (let i = 0; i < shots; i++) {
          this.state.real.dispose();
          this.state.imag.dispose();
          this.state = { real: cachedReal.clone(), imag: cachedImag.clone() };
          this._runCircuit(this.state, mcmIdx);
          this._sampleAndCount(1);
        }

        cachedReal.dispose();
        cachedImag.dispose();
      } else {
        for (let i = 0; i < shots; i++) {
          this._runCircuit();
          this._sampleAndCount(1);
        }
      }
    } else {
      // No MCM - run once and sample
      this._runCircuit();
      this._sampleAndCount(shots);
    }

    return this;
  }

  /**
   * Find index of first mid-circuit measurement.
   * @private
   */
  _findFirstMCM() {
    for (let i = 0; i < this.circuit.length; i++) {
      if (this.circuit[i].name === 'MCM') {
        return i;
      }
    }
    return null;
  }

  /**
   * Run the circuit to compute statevector.
   * @private
   */
  _runCircuit(state = null, startIdx = 0, stopIdx = null) {
    if (stopIdx === null) stopIdx = this.circuit.length;

    // Initialize state
    if (state === null) {
      if (this.state) {
        this.state.real.dispose();
        this.state.imag.dispose();
      }

      const realData = new Array(2 ** this.size).fill(0);
      realData[0] = 1; // |0...0> state
      const imagData = new Array(2 ** this.size).fill(0);

      this.state = {
        real: tf.tensor1d(realData),
        imag: tf.tensor1d(imagData)
      };
    } else {
      this.state = state;
    }

    // Apply gates
    for (let i = startIdx; i < stopIdx; i++) {
      this._applyGate(this.circuit[i]);
    }
  }

  /**
   * Apply a single gate operation.
   * @private
   */
  _applyGate(gateMeta) {
    const { name, target, controls, angles } = gateMeta;

    if (name === 'BARRIER') return;

    // Handle two-qubit gates
    if (name === 'CNOT') {
      if (!controls || controls.length === 0) {
        throw new Error('CNOT requires a control qubit');
      }
      this.gateObj.CNOT(this.state, controls[0], target);
      return;
    }

    if (name === 'CZ') {
      if (!controls || controls.length === 0) {
        throw new Error('CZ requires a control qubit');
      }
      this.gateObj.CZ(this.state, controls[0], target);
      return;
    }

    if (name === 'SWAP') {
      if (!controls || controls.length === 0) {
        throw new Error('SWAP requires two qubits');
      }
      this.gateObj.SWAP(this.state, controls[0], target);
      return;
    }

    // Handle MCM
    if (name === 'MCM') {
      this.gateObj.measure(this.state, target);
      return;
    }

    // Get gate matrix
    const gateEntry = this._gateRegistry[name];
    let gateMatrix;

    if (typeof gateEntry === 'function') {
      if (angles === undefined) {
        throw new Error(`Gate '${name}' requires angle parameter(s)`);
      }
      if (Array.isArray(angles)) {
        gateMatrix = gateEntry(...angles);
      } else {
        gateMatrix = gateEntry(angles);
      }
    } else {
      gateMatrix = gateEntry;
    }

    // Apply gate
    this.gateObj.apply(gateMatrix, this.state, target, controls);
  }

  /**
   * Sample from statevector and update measurement counts.
   * @private
   */
  _sampleAndCount(shots) {
    const realData = this.state.real.arraySync();
    const imagData = this.state.imag.arraySync();
    const N = realData.length;

    // Compute probabilities
    const probs = new Array(N);
    for (let i = 0; i < N; i++) {
      probs[i] = realData[i] * realData[i] + imagData[i] * imagData[i];
    }

    // Compute CDF
    const cdf = new Array(N);
    cdf[0] = probs[0];
    for (let i = 1; i < N; i++) {
      cdf[i] = cdf[i - 1] + probs[i];
    }

    // Sample
    for (let s = 0; s < shots; s++) {
      const r = Math.random();
      // Binary search in CDF
      let lo = 0, hi = N - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cdf[mid] < r) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      this.measurements[lo]++;
    }
  }

  // ===========================================================================
  // Results and Analysis
  // ===========================================================================

  /**
   * Get the final statevector.
   * @returns {{real: number[], imag: number[]}|null}
   */
  getStatevector() {
    if (!this.state) return null;
    return {
      real: this.state.real.arraySync(),
      imag: this.state.imag.arraySync()
    };
  }

  /**
   * Get probability distribution.
   * @returns {number[]|null}
   */
  getProbabilities() {
    if (!this.state) return null;
    const real = this.state.real.arraySync();
    const imag = this.state.imag.arraySync();
    return real.map((r, i) => r * r + imag[i] * imag[i]);
  }

  /**
   * Get measurement counts as an object.
   * @returns {Object<string, number>|null}
   */
  getCounts() {
    if (!this.measurements) return null;

    const counts = {};
    for (let i = 0; i < this.measurements.length; i++) {
      if (this.measurements[i] > 0) {
        // Little-endian bitstring
        const bitstring = i.toString(2).padStart(this.size, '0').split('').reverse().join('');
        counts[bitstring] = this.measurements[i];
      }
    }
    return counts;
  }

  /**
   * Get measurement counts with big-endian bitstrings.
   * @returns {Object<string, number>|null}
   */
  getCountsBigEndian() {
    if (!this.measurements) return null;

    const counts = {};
    for (let i = 0; i < this.measurements.length; i++) {
      if (this.measurements[i] > 0) {
        const bitstring = i.toString(2).padStart(this.size, '0');
        counts[bitstring] = this.measurements[i];
      }
    }
    return counts;
  }

  // ===========================================================================
  // Circuit Information
  // ===========================================================================

  /**
   * Calculate circuit depth.
   * @returns {number}
   */
  depth() {
    if (this.circuit.length === 0) return 0;

    const qubitDepth = new Array(this.size).fill(0);

    for (const gate of this.circuit) {
      if (gate.name === 'BARRIER') continue;

      const involved = [gate.target, ...(gate.controls || [])];
      const maxDepth = Math.max(...involved.map(q => qubitDepth[q]));
      const newDepth = maxDepth + 1;

      for (const q of involved) {
        qubitDepth[q] = newDepth;
      }
    }

    return Math.max(...qubitDepth);
  }

  /**
   * Count occurrences of each gate type.
   * @returns {Object<string, number>}
   */
  gateCount() {
    const counts = {};
    for (const gate of this.circuit) {
      if (gate.name === 'BARRIER') continue;
      counts[gate.name] = (counts[gate.name] || 0) + 1;
    }
    return counts;
  }

  /**
   * Get number of gates (excluding barriers).
   * @returns {number}
   */
  get length() {
    return this.circuit.filter(g => g.name !== 'BARRIER').length;
  }

  /**
   * String representation.
   * @returns {string}
   */
  toString() {
    return `Circuit(qubits=${this.size}, gates=${this.length}, depth=${this.depth()})`;
  }

  /**
   * Print circuit structure.
   */
  printCircuit() {
    console.log(`Circuit with ${this.size} qubits:`);
    console.log('-'.repeat(40));

    for (let i = 0; i < this.circuit.length; i++) {
      const gate = this.circuit[i];
      if (gate.name === 'BARRIER') {
        console.log(`  [${i}] ──── BARRIER ────`);
        continue;
      }

      let line = `  [${i}] ${gate.name} @ q${gate.target}`;

      if (gate.controls) {
        const ctrlStr = gate.controls.map(c => `q${c}`).join(', ');
        line += ` (ctrl: ${ctrlStr})`;
      }

      if (gate.angles !== undefined) {
        const angleStr = Array.isArray(gate.angles)
          ? gate.angles.map(a => a.toFixed(4)).join(', ')
          : gate.angles.toFixed(4);
        line += ` (angles: ${angleStr})`;
      }

      console.log(line);
    }
    console.log('-'.repeat(40));
  }

  /**
   * Reset circuit and state.
   * @returns {Circuit}
   */
  reset() {
    this.circuit = [];
    if (this.state) {
      this.state.real.dispose();
      this.state.imag.dispose();
      this.state = null;
    }
    this.measurements = null;
    return this;
  }

  /**
   * Create a copy of this circuit.
   * @returns {Circuit}
   */
  copy() {
    const newCircuit = new Circuit(this.size, { device: this.device });
    newCircuit.circuit = this.circuit.map(g => ({ ...g }));
    return newCircuit;
  }

  /**
   * Dispose of tensor resources.
   */
  dispose() {
    if (this.state) {
      this.state.real.dispose();
      this.state.imag.dispose();
      this.state = null;
    }
  }
}

export default Circuit;
