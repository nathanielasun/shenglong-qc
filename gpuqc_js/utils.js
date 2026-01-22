/**
 * Utility Functions for Quantum Circuit Simulation.
 *
 * This module provides utility functions for:
 * - Memory monitoring
 * - Statevector I/O
 * - Device detection
 * - Analysis functions
 *
 * @author Nathaniel Sun
 */

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// Memory Monitoring
// =============================================================================

/**
 * Get current memory usage from TensorFlow.js.
 * @returns {{numTensors: number, numBytes: number, numDataBuffers: number}}
 */
export function getMemoryInfo() {
  return tf.memory();
}

/**
 * Get memory usage in megabytes.
 * @returns {number}
 */
export function getMemoryUsageMB() {
  const mem = tf.memory();
  return mem.numBytes / (1024 * 1024);
}

/**
 * Print memory statistics.
 */
export function printMemoryStats() {
  const mem = tf.memory();
  console.log(`[TensorFlow.js Memory]`);
  console.log(`  Tensors: ${mem.numTensors}`);
  console.log(`  Data buffers: ${mem.numDataBuffers}`);
  console.log(`  Bytes: ${(mem.numBytes / (1024 * 1024)).toFixed(2)} MB`);

  if (mem.numBytesInGPU !== undefined) {
    console.log(`  GPU Bytes: ${(mem.numBytesInGPU / (1024 * 1024)).toFixed(2)} MB`);
  }
}

/**
 * Estimate memory required for a statevector.
 * @param {number} numQubits - Number of qubits
 * @returns {number} Estimated memory in MB
 */
export function estimateStatevectorMemory(numQubits) {
  const numAmplitudes = 2 ** numQubits;
  // Each amplitude: 2 float32 values (real + imag) = 8 bytes
  const totalBytes = numAmplitudes * 8;
  return totalBytes / (1024 * 1024);
}

// =============================================================================
// Device Utilities
// =============================================================================

/**
 * Get available TensorFlow.js backends.
 * @returns {Promise<Object<string, boolean>>}
 */
export async function getAvailableDevices() {
  const devices = {
    cpu: true,
    webgl: false,
    wasm: false,
    tensorflow: false,
  };

  // Check WebGL
  try {
    const canvas = typeof document !== 'undefined'
      ? document.createElement('canvas')
      : null;
    if (canvas) {
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      devices.webgl = !!gl;
    }
  } catch {
    devices.webgl = false;
  }

  // Check native TensorFlow (CUDA/Metal)
  try {
    // This will only work in Node.js with tfjs-node installed
    const tfjsNode = await import('@tensorflow/tfjs-node').catch(() => null);
    if (tfjsNode) {
      devices.tensorflow = true;
    }
  } catch {
    devices.tensorflow = false;
  }

  // Check GPU variant
  try {
    const tfjsNodeGpu = await import('@tensorflow/tfjs-node-gpu').catch(() => null);
    if (tfjsNodeGpu) {
      devices.tensorflowGpu = true;
    }
  } catch {
    devices.tensorflowGpu = false;
  }

  return devices;
}

/**
 * Get the current active backend.
 * @returns {string}
 */
export function getCurrentBackend() {
  return tf.getBackend();
}

/**
 * Get detailed backend information.
 * @returns {Object}
 */
export function getBackendInfo() {
  const backend = tf.getBackend();
  const mem = tf.memory();

  return {
    backend,
    memory: {
      numTensors: mem.numTensors,
      numBytes: mem.numBytes,
      numBytesInGPU: mem.numBytesInGPU,
    },
  };
}

// =============================================================================
// Statevector I/O
// =============================================================================

/**
 * Save statevector to a JSON file.
 * @param {{real: number[], imag: number[]}} state - Statevector
 * @param {string} filename - Output filename
 */
export function saveStatevector(state, filename) {
  const data = JSON.stringify({
    real: Array.isArray(state.real) ? state.real : state.real.arraySync(),
    imag: Array.isArray(state.imag) ? state.imag : state.imag.arraySync(),
  });

  fs.writeFileSync(filename, data);
  console.log(`Saved statevector to ${filename}`);
}

/**
 * Load statevector from a JSON file.
 * @param {string} filename - Input filename
 * @returns {{real: number[], imag: number[]}}
 */
export function loadStatevector(filename) {
  const data = fs.readFileSync(filename, 'utf-8');
  const parsed = JSON.parse(data);
  return {
    real: parsed.real,
    imag: parsed.imag,
  };
}

/**
 * Save statevector to binary format.
 * @param {{real: number[], imag: number[]}} state - Statevector
 * @param {string} filename - Output filename
 */
export function saveStatevectorBinary(state, filename) {
  const real = Array.isArray(state.real) ? state.real : state.real.arraySync();
  const imag = Array.isArray(state.imag) ? state.imag : state.imag.arraySync();

  const N = real.length;
  const buffer = Buffer.alloc(N * 8); // 2 float32 per amplitude

  for (let i = 0; i < N; i++) {
    buffer.writeFloatLE(real[i], i * 8);
    buffer.writeFloatLE(imag[i], i * 8 + 4);
  }

  fs.writeFileSync(filename, buffer);
  console.log(`Saved statevector to ${filename} (binary format)`);
}

/**
 * Load statevector from binary format.
 * @param {string} filename - Input filename
 * @returns {{real: number[], imag: number[]}}
 */
export function loadStatevectorBinary(filename) {
  const buffer = fs.readFileSync(filename);
  const N = buffer.length / 8;

  const real = new Array(N);
  const imag = new Array(N);

  for (let i = 0; i < N; i++) {
    real[i] = buffer.readFloatLE(i * 8);
    imag[i] = buffer.readFloatLE(i * 8 + 4);
  }

  return { real, imag };
}

// =============================================================================
// Analysis Functions
// =============================================================================

/**
 * Calculate fidelity between two pure states.
 * F = |<psi1|psi2>|^2
 *
 * @param {{real: number[], imag: number[]}} state1
 * @param {{real: number[], imag: number[]}} state2
 * @returns {number} Fidelity value between 0 and 1
 */
export function fidelity(state1, state2) {
  const real1 = Array.isArray(state1.real) ? state1.real : state1.real.arraySync();
  const imag1 = Array.isArray(state1.imag) ? state1.imag : state1.imag.arraySync();
  const real2 = Array.isArray(state2.real) ? state2.real : state2.real.arraySync();
  const imag2 = Array.isArray(state2.imag) ? state2.imag : state2.imag.arraySync();

  if (real1.length !== real2.length) {
    throw new Error('Statevectors must have the same length');
  }

  // Compute <psi1|psi2> = sum(conj(psi1) * psi2)
  let innerReal = 0;
  let innerImag = 0;

  for (let i = 0; i < real1.length; i++) {
    // conj(a+bi) * (c+di) = (ac+bd) + (ad-bc)i
    innerReal += real1[i] * real2[i] + imag1[i] * imag2[i];
    innerImag += real1[i] * imag2[i] - imag1[i] * real2[i];
  }

  // |inner|^2
  return innerReal * innerReal + innerImag * innerImag;
}

/**
 * Calculate trace distance between two pure states.
 * D = sqrt(1 - F)
 *
 * @param {{real: number[], imag: number[]}} state1
 * @param {{real: number[], imag: number[]}} state2
 * @returns {number}
 */
export function traceDistance(state1, state2) {
  const f = fidelity(state1, state2);
  return Math.sqrt(1 - f);
}

/**
 * Get non-zero amplitudes from a statevector.
 * @param {{real: number[], imag: number[]}} state
 * @param {number} threshold - Minimum magnitude
 * @param {number} numQubits - Number of qubits (inferred if not provided)
 * @returns {Object<string, {re: number, im: number}>}
 */
export function getNonzeroAmplitudes(state, threshold = 1e-10, numQubits = null) {
  const real = Array.isArray(state.real) ? state.real : state.real.arraySync();
  const imag = Array.isArray(state.imag) ? state.imag : state.imag.arraySync();

  if (numQubits === null) {
    numQubits = Math.log2(real.length);
  }

  const result = {};

  for (let i = 0; i < real.length; i++) {
    const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
    if (mag > threshold) {
      const bitstring = i.toString(2).padStart(numQubits, '0');
      result[bitstring] = { re: real[i], im: imag[i] };
    }
  }

  return result;
}

/**
 * Print statevector in human-readable format.
 * @param {{real: number[], imag: number[]}} state
 * @param {Object} options
 * @param {number} options.threshold - Minimum magnitude to display
 * @param {number} options.maxTerms - Maximum terms to show
 * @param {number} options.precision - Decimal places
 */
export function printStatevector(state, options = {}) {
  const { threshold = 1e-10, maxTerms = 20, precision = 4 } = options;

  const real = Array.isArray(state.real) ? state.real : state.real.arraySync();
  const imag = Array.isArray(state.imag) ? state.imag : state.imag.arraySync();
  const numQubits = Math.log2(real.length);

  const amps = getNonzeroAmplitudes(state, threshold, numQubits);
  const sorted = Object.entries(amps)
    .map(([basis, amp]) => ({
      basis,
      amp,
      prob: amp.re * amp.re + amp.im * amp.im
    }))
    .sort((a, b) => b.prob - a.prob);

  console.log(`Statevector (${numQubits} qubits, ${sorted.length} non-zero terms):`);
  console.log('-'.repeat(40));

  const toShow = sorted.slice(0, maxTerms);

  for (const { basis, amp, prob } of toShow) {
    let ampStr;
    if (Math.abs(amp.im) < 1e-10) {
      ampStr = amp.re >= 0 ? `+${amp.re.toFixed(precision)}` : amp.re.toFixed(precision);
    } else if (Math.abs(amp.re) < 1e-10) {
      ampStr = amp.im >= 0 ? `+${amp.im.toFixed(precision)}i` : `${amp.im.toFixed(precision)}i`;
    } else {
      const reStr = amp.re >= 0 ? `+${amp.re.toFixed(precision)}` : amp.re.toFixed(precision);
      const imStr = amp.im >= 0 ? `+${amp.im.toFixed(precision)}i` : `${amp.im.toFixed(precision)}i`;
      ampStr = `(${reStr}${imStr})`;
    }

    console.log(`  |${basis}> : ${ampStr}  (p=${prob.toFixed(precision)})`);
  }

  if (sorted.length > maxTerms) {
    console.log(`  ... and ${sorted.length - maxTerms} more terms`);
  }

  console.log('-'.repeat(40));
}

// =============================================================================
// Measurement Utilities
// =============================================================================

/**
 * Convert measurement counts to probabilities.
 * @param {Object<string, number>} counts
 * @returns {Object<string, number>}
 */
export function countsToProbabilities(counts) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const probs = {};
  for (const [state, count] of Object.entries(counts)) {
    probs[state] = count / total;
  }
  return probs;
}

/**
 * Get the most likely measurement outcomes.
 * @param {Object<string, number>} counts
 * @param {number} n - Number of top outcomes
 * @returns {Array<{state: string, count: number, probability: number}>}
 */
export function getTopOutcomes(counts, n = 5) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return Object.entries(counts)
    .map(([state, count]) => ({
      state,
      count,
      probability: count / total
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, n);
}

/**
 * Print measurement results in a formatted way.
 * @param {Object<string, number>} counts
 * @param {Object} options
 * @param {number} options.maxStates - Maximum states to display
 */
export function printCounts(counts, options = {}) {
  const { maxStates = 32 } = options;

  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const sorted = Object.entries(counts)
    .map(([state, count]) => ({ state, count, prob: count / total }))
    .sort((a, b) => b.count - a.count);

  console.log('\nMeasurement Results:');
  console.log('='.repeat(50));

  const toShow = sorted.slice(0, maxStates);
  const maxCount = Math.max(...toShow.map(x => x.count));
  const barWidth = 30;

  for (const { state, count, prob } of toShow) {
    const barLen = Math.round((count / maxCount) * barWidth);
    const bar = '#'.repeat(barLen);
    console.log(`  |${state}>: ${count.toString().padStart(6)} (${(prob * 100).toFixed(1).padStart(5)}%) ${bar}`);
  }

  if (sorted.length > maxStates) {
    console.log(`  ... and ${sorted.length - maxStates} more states`);
  }

  console.log('='.repeat(50));
  console.log(`Total shots: ${total}`);
}

export default {
  getMemoryInfo,
  getMemoryUsageMB,
  printMemoryStats,
  estimateStatevectorMemory,
  getAvailableDevices,
  getCurrentBackend,
  getBackendInfo,
  saveStatevector,
  loadStatevector,
  saveStatevectorBinary,
  loadStatevectorBinary,
  fidelity,
  traceDistance,
  getNonzeroAmplitudes,
  printStatevector,
  countsToProbabilities,
  getTopOutcomes,
  printCounts,
};
