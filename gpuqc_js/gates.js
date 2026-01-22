/**
 * Quantum Gate Definitions for Statevector Simulation.
 *
 * This module provides the Gate class which contains all standard quantum gates
 * and gate application routines for statevector simulation. Supports CPU, CUDA,
 * and Metal backends via TensorFlow.js.
 *
 * Complex numbers are represented as objects {re: number, im: number} or
 * as paired tensors [realTensor, imagTensor] for GPU operations.
 *
 * @author Nathaniel Sun
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Complex number utilities for quantum operations.
 */
export const Complex = {
  /**
   * Create a complex number.
   * @param {number} re - Real part
   * @param {number} im - Imaginary part
   * @returns {{re: number, im: number}}
   */
  create(re, im = 0) {
    return { re, im };
  },

  /**
   * Multiply two complex numbers.
   * @param {{re: number, im: number}} a
   * @param {{re: number, im: number}} b
   * @returns {{re: number, im: number}}
   */
  mul(a, b) {
    return {
      re: a.re * b.re - a.im * b.im,
      im: a.re * b.im + a.im * b.re
    };
  },

  /**
   * Add two complex numbers.
   * @param {{re: number, im: number}} a
   * @param {{re: number, im: number}} b
   * @returns {{re: number, im: number}}
   */
  add(a, b) {
    return { re: a.re + b.re, im: a.im + b.im };
  },

  /**
   * Complex exponential e^(i*theta).
   * @param {number} theta - Angle in radians
   * @returns {{re: number, im: number}}
   */
  exp(theta) {
    return { re: Math.cos(theta), im: Math.sin(theta) };
  },

  /**
   * Absolute value squared |z|^2.
   * @param {{re: number, im: number}} z
   * @returns {number}
   */
  abs2(z) {
    return z.re * z.re + z.im * z.im;
  },

  /**
   * Absolute value |z|.
   * @param {{re: number, im: number}} z
   * @returns {number}
   */
  abs(z) {
    return Math.sqrt(this.abs2(z));
  }
};

/**
 * A collection of quantum gates and gate application methods.
 *
 * This class provides standard single-qubit gates, rotation gates,
 * controlled gates, and measurement operations for quantum circuit
 * simulation using statevector representation.
 *
 * @example
 * const gate = new Gate();
 * const state = { real: tf.tensor1d([1, 0]), imag: tf.tensor1d([0, 0]) };
 * gate.apply(gate.H, state, 0);
 */
export class Gate {
  /**
   * Initialize the Gate object with standard quantum gates.
   */
  constructor() {
    this._initializeGates();
  }

  /**
   * Initialize all standard quantum gate matrices.
   * Gates are stored as 2x2 arrays of complex numbers.
   * @private
   */
  _initializeGates() {
    const C = Complex.create;
    const sqrt2inv = 1 / Math.sqrt(2);

    // Projection operators
    this.zero = [
      [C(1), C(0)],
      [C(0), C(0)]
    ];

    this.one = [
      [C(0), C(0)],
      [C(0), C(1)]
    ];

    // Hadamard gate: creates superposition
    // H|0> = |+> = (|0> + |1>)/sqrt(2)
    this.H = [
      [C(sqrt2inv), C(sqrt2inv)],
      [C(sqrt2inv), C(-sqrt2inv)]
    ];

    // Identity gate
    this.I = [
      [C(1), C(0)],
      [C(0), C(1)]
    ];

    // Pauli-X gate (NOT gate, bit flip)
    this.X = [
      [C(0), C(1)],
      [C(1), C(0)]
    ];

    // Pauli-Y gate
    this.Y = [
      [C(0), C(0, -1)],
      [C(0, 1), C(0)]
    ];

    // Pauli-Z gate (phase flip)
    this.Z = [
      [C(1), C(0)],
      [C(0), C(-1)]
    ];

    // S gate (sqrt(Z), phase gate with pi/2 rotation)
    this.S = [
      [C(1), C(0)],
      [C(0), C(0, 1)]
    ];

    // S-dagger gate (inverse of S)
    this.Sdg = [
      [C(1), C(0)],
      [C(0), C(0, -1)]
    ];

    // T gate (sqrt(S), fourth root of Z)
    this.T = this.P(Math.PI / 4);

    // T-dagger gate (inverse of T)
    this.Tdg = this.P(-Math.PI / 4);

    // sqrt(X) gate
    this.SX = [
      [C(0.5, 0.5), C(0.5, -0.5)],
      [C(0.5, -0.5), C(0.5, 0.5)]
    ];
  }

  // ===========================================================================
  // Single-Qubit Gate Application
  // ===========================================================================

  /**
   * Apply a single-qubit gate to the statevector using strided indexing.
   *
   * This method efficiently applies a 2x2 gate matrix to a specific qubit
   * in the statevector without constructing large tensor products.
   *
   * @param {Array<Array<{re: number, im: number}>>} gate - 2x2 gate matrix
   * @param {{real: tf.Tensor1D, imag: tf.Tensor1D}} state - Statevector
   * @param {number} target - Target qubit index (0-indexed from LSB)
   * @param {number[]|null} controls - Optional control qubit indices
   */
  apply(gate, state, target, controls = null) {
    const N = state.real.shape[0];
    const numQubits = Math.log2(N);

    if (target < 0) {
      throw new Error('Target qubit index cannot be negative');
    }
    if (target >= numQubits) {
      throw new Error(`Target qubit ${target} out of range for ${numQubits}-qubit system`);
    }

    const targetBit = 1 << target;

    // Generate indices where target bit = 0
    let idx0 = [];
    for (let i = 0; i < N; i++) {
      if ((i & targetBit) === 0) {
        if (controls) {
          // Check all control bits are 1
          const controlMask = controls.reduce((acc, c) => acc | (1 << c), 0);
          if ((i & controlMask) === controlMask) {
            idx0.push(i);
          }
        } else {
          idx0.push(i);
        }
      }
    }

    // Corresponding indices where target bit = 1
    const idx1 = idx0.map(i => i | targetBit);

    // Get current amplitudes
    const realData = state.real.arraySync();
    const imagData = state.imag.arraySync();

    // Apply gate transformation
    for (let k = 0; k < idx0.length; k++) {
      const i0 = idx0[k];
      const i1 = idx1[k];

      const v0 = { re: realData[i0], im: imagData[i0] };
      const v1 = { re: realData[i1], im: imagData[i1] };

      // out0 = gate[0][0] * v0 + gate[0][1] * v1
      const t00 = Complex.mul(gate[0][0], v0);
      const t01 = Complex.mul(gate[0][1], v1);
      const out0 = Complex.add(t00, t01);

      // out1 = gate[1][0] * v0 + gate[1][1] * v1
      const t10 = Complex.mul(gate[1][0], v0);
      const t11 = Complex.mul(gate[1][1], v1);
      const out1 = Complex.add(t10, t11);

      realData[i0] = out0.re;
      imagData[i0] = out0.im;
      realData[i1] = out1.re;
      imagData[i1] = out1.im;
    }

    // Update state tensors
    state.real.dispose();
    state.imag.dispose();
    state.real = tf.tensor1d(realData);
    state.imag = tf.tensor1d(imagData);
  }

  /**
   * Apply gate using GPU-accelerated tensor operations.
   * More efficient for larger statevectors.
   *
   * @param {Array<Array<{re: number, im: number}>>} gate - 2x2 gate matrix
   * @param {{real: tf.Tensor1D, imag: tf.Tensor1D}} state - Statevector
   * @param {number} target - Target qubit index
   * @param {number[]|null} controls - Optional control qubit indices
   */
  applyGPU(gate, state, target, controls = null) {
    const N = state.real.shape[0];
    const targetBit = 1 << target;

    // Build index arrays
    const idx0 = [];
    const idx1 = [];

    for (let i = 0; i < N; i++) {
      if ((i & targetBit) === 0) {
        if (controls) {
          const controlMask = controls.reduce((acc, c) => acc | (1 << c), 0);
          if ((i & controlMask) === controlMask) {
            idx0.push(i);
            idx1.push(i | targetBit);
          }
        } else {
          idx0.push(i);
          idx1.push(i | targetBit);
        }
      }
    }

    const idx0Tensor = tf.tensor1d(idx0, 'int32');
    const idx1Tensor = tf.tensor1d(idx1, 'int32');

    // Gather amplitudes
    const v0Real = tf.gather(state.real, idx0Tensor);
    const v0Imag = tf.gather(state.imag, idx0Tensor);
    const v1Real = tf.gather(state.real, idx1Tensor);
    const v1Imag = tf.gather(state.imag, idx1Tensor);

    // Gate matrix elements
    const g00 = gate[0][0], g01 = gate[0][1];
    const g10 = gate[1][0], g11 = gate[1][1];

    // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // out0 = g00 * v0 + g01 * v1
    const out0Real = tf.add(
      tf.sub(tf.mul(v0Real, g00.re), tf.mul(v0Imag, g00.im)),
      tf.sub(tf.mul(v1Real, g01.re), tf.mul(v1Imag, g01.im))
    );
    const out0Imag = tf.add(
      tf.add(tf.mul(v0Real, g00.im), tf.mul(v0Imag, g00.re)),
      tf.add(tf.mul(v1Real, g01.im), tf.mul(v1Imag, g01.re))
    );

    // out1 = g10 * v0 + g11 * v1
    const out1Real = tf.add(
      tf.sub(tf.mul(v0Real, g10.re), tf.mul(v0Imag, g10.im)),
      tf.sub(tf.mul(v1Real, g11.re), tf.mul(v1Imag, g11.im))
    );
    const out1Imag = tf.add(
      tf.add(tf.mul(v0Real, g10.im), tf.mul(v0Imag, g10.re)),
      tf.add(tf.mul(v1Real, g11.im), tf.mul(v1Imag, g11.re))
    );

    // Scatter back using tensorScatterUpdate
    const allIdx = tf.concat([idx0Tensor, idx1Tensor]);
    const allReal = tf.concat([out0Real, out1Real]);
    const allImag = tf.concat([out0Imag, out1Imag]);

    const newReal = tf.tensorScatterUpdate(
      state.real,
      allIdx.reshape([-1, 1]),
      allReal
    );
    const newImag = tf.tensorScatterUpdate(
      state.imag,
      allIdx.reshape([-1, 1]),
      allImag
    );

    // Cleanup old tensors
    state.real.dispose();
    state.imag.dispose();
    v0Real.dispose(); v0Imag.dispose();
    v1Real.dispose(); v1Imag.dispose();
    out0Real.dispose(); out0Imag.dispose();
    out1Real.dispose(); out1Imag.dispose();
    idx0Tensor.dispose(); idx1Tensor.dispose();
    allIdx.dispose(); allReal.dispose(); allImag.dispose();

    state.real = newReal;
    state.imag = newImag;
  }

  // ===========================================================================
  // Rotation Gates
  // ===========================================================================

  /**
   * Create a global phase gate.
   * @param {number} phase - Phase angle in radians
   * @returns {Array<Array<{re: number, im: number}>>}
   */
  Ph(phase) {
    const exp = Complex.exp(phase);
    return [
      [exp, Complex.create(0)],
      [Complex.create(0), exp]
    ];
  }

  /**
   * Create a phase shift gate P(phi).
   * P(phi)|0> = |0>, P(phi)|1> = e^(i*phi)|1>
   * @param {number} phi - Phase angle in radians
   * @returns {Array<Array<{re: number, im: number}>>}
   */
  P(phi) {
    return [
      [Complex.create(1), Complex.create(0)],
      [Complex.create(0), Complex.exp(phi)]
    ];
  }

  /**
   * Create an X-axis rotation gate Rx(theta).
   * @param {number} angle - Rotation angle in radians
   * @returns {Array<Array<{re: number, im: number}>>}
   */
  Rx(angle) {
    const cosT = Math.cos(angle / 2);
    const sinT = Math.sin(angle / 2);
    return [
      [Complex.create(cosT), Complex.create(0, -sinT)],
      [Complex.create(0, -sinT), Complex.create(cosT)]
    ];
  }

  /**
   * Create a Y-axis rotation gate Ry(theta).
   * @param {number} angle - Rotation angle in radians
   * @returns {Array<Array<{re: number, im: number}>>}
   */
  Ry(angle) {
    const cosT = Math.cos(angle / 2);
    const sinT = Math.sin(angle / 2);
    return [
      [Complex.create(cosT), Complex.create(-sinT)],
      [Complex.create(sinT), Complex.create(cosT)]
    ];
  }

  /**
   * Create a Z-axis rotation gate Rz(theta).
   * @param {number} angle - Rotation angle in radians
   * @returns {Array<Array<{re: number, im: number}>>}
   */
  Rz(angle) {
    return [
      [Complex.exp(-angle / 2), Complex.create(0)],
      [Complex.create(0), Complex.exp(angle / 2)]
    ];
  }

  /**
   * Create a general single-qubit unitary gate U(theta, phi, lambda).
   * @param {number} theta - Rotation angle
   * @param {number} phi - Phase angle
   * @param {number} lam - Lambda angle
   * @returns {Array<Array<{re: number, im: number}>>}
   */
  U(theta, phi, lam) {
    const cosT = Math.cos(theta / 2);
    const sinT = Math.sin(theta / 2);
    const expLam = Complex.exp(lam);
    const expPhi = Complex.exp(phi);
    const expPhiLam = Complex.exp(phi + lam);

    return [
      [
        Complex.create(cosT),
        Complex.mul(Complex.create(-1), Complex.mul(expLam, Complex.create(sinT)))
      ],
      [
        Complex.mul(expPhi, Complex.create(sinT)),
        Complex.mul(expPhiLam, Complex.create(cosT))
      ]
    ];
  }

  // ===========================================================================
  // Two-Qubit Gates
  // ===========================================================================

  /**
   * Apply a CNOT (controlled-X) gate.
   * @param {{real: tf.Tensor1D, imag: tf.Tensor1D}} state - Statevector
   * @param {number} control - Control qubit index
   * @param {number} target - Target qubit index
   */
  CNOT(state, control, target) {
    const N = state.real.shape[0];
    const numQubits = Math.log2(N);

    if (control < 0 || target < 0) {
      throw new Error('Qubit indices cannot be negative');
    }
    if (control >= numQubits || target >= numQubits) {
      throw new Error(`Qubit indices out of range for ${numQubits}-qubit system`);
    }
    if (control === target) {
      throw new Error('Control and target qubits must be different');
    }

    const controlBit = 1 << control;
    const targetBit = 1 << target;

    const realData = state.real.arraySync();
    const imagData = state.imag.arraySync();

    // Find indices where control is |1> and swap with target flipped
    for (let i = 0; i < N; i++) {
      if ((i & controlBit) !== 0) {
        const j = i ^ targetBit;
        if (i < j) {
          // Swap amplitudes
          [realData[i], realData[j]] = [realData[j], realData[i]];
          [imagData[i], imagData[j]] = [imagData[j], imagData[i]];
        }
      }
    }

    state.real.dispose();
    state.imag.dispose();
    state.real = tf.tensor1d(realData);
    state.imag = tf.tensor1d(imagData);
  }

  /**
   * Apply a CZ (controlled-Z) gate.
   * @param {{real: tf.Tensor1D, imag: tf.Tensor1D}} state - Statevector
   * @param {number} control - Control qubit index
   * @param {number} target - Target qubit index
   */
  CZ(state, control, target) {
    const N = state.real.shape[0];
    const bothOneMask = (1 << control) | (1 << target);

    const realData = state.real.arraySync();
    const imagData = state.imag.arraySync();

    // Apply phase flip to |11> states
    for (let i = 0; i < N; i++) {
      if ((i & bothOneMask) === bothOneMask) {
        realData[i] = -realData[i];
        imagData[i] = -imagData[i];
      }
    }

    state.real.dispose();
    state.imag.dispose();
    state.real = tf.tensor1d(realData);
    state.imag = tf.tensor1d(imagData);
  }

  /**
   * Apply a SWAP gate.
   * @param {{real: tf.Tensor1D, imag: tf.Tensor1D}} state - Statevector
   * @param {number} qubit1 - First qubit index
   * @param {number} qubit2 - Second qubit index
   */
  SWAP(state, qubit1, qubit2) {
    if (qubit1 === qubit2) return;

    const N = state.real.shape[0];
    const bit1 = 1 << qubit1;
    const bit2 = 1 << qubit2;

    const realData = state.real.arraySync();
    const imagData = state.imag.arraySync();

    // Swap amplitudes where exactly one of the qubits is |1>
    for (let i = 0; i < N; i++) {
      const has1 = (i & bit1) !== 0;
      const has2 = (i & bit2) !== 0;

      if (has1 !== has2) {
        // Compute swapped index
        const j = (i ^ bit1) ^ bit2;
        if (i < j) {
          [realData[i], realData[j]] = [realData[j], realData[i]];
          [imagData[i], imagData[j]] = [imagData[j], imagData[i]];
        }
      }
    }

    state.real.dispose();
    state.imag.dispose();
    state.real = tf.tensor1d(realData);
    state.imag = tf.tensor1d(imagData);
  }

  // ===========================================================================
  // Measurement
  // ===========================================================================

  /**
   * Perform a mid-circuit measurement on a single qubit.
   * Collapses the statevector and returns the measurement result.
   *
   * @param {{real: tf.Tensor1D, imag: tf.Tensor1D}} state - Statevector
   * @param {number} target - Qubit to measure
   * @returns {number} Measurement result (0 or 1)
   */
  measure(state, target) {
    const N = state.real.shape[0];
    const numQubits = Math.log2(N);

    if (target < 0 || target >= numQubits) {
      throw new Error(`Target qubit ${target} out of range`);
    }

    const targetBit = 1 << target;
    const realData = state.real.arraySync();
    const imagData = state.imag.arraySync();

    // Compute probability of measuring |1>
    let probOne = 0;
    for (let i = 0; i < N; i++) {
      if ((i & targetBit) !== 0) {
        probOne += realData[i] * realData[i] + imagData[i] * imagData[i];
      }
    }

    // Sample measurement outcome
    const outcome = Math.random() < probOne ? 1 : 0;

    // Collapse and renormalize
    const norm = outcome === 1 ? Math.sqrt(probOne) : Math.sqrt(1 - probOne);

    for (let i = 0; i < N; i++) {
      const inOneSubspace = (i & targetBit) !== 0;

      if ((outcome === 1 && !inOneSubspace) || (outcome === 0 && inOneSubspace)) {
        realData[i] = 0;
        imagData[i] = 0;
      } else if (norm > 0) {
        realData[i] /= norm;
        imagData[i] /= norm;
      }
    }

    state.real.dispose();
    state.imag.dispose();
    state.real = tf.tensor1d(realData);
    state.imag = tf.tensor1d(imagData);

    return outcome;
  }
}

export default Gate;
