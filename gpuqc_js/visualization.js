/**
 * Circuit Visualization Module.
 *
 * This module provides ASCII circuit diagram generation for terminal display
 * and probability visualization for quantum circuits.
 *
 * @author Nathaniel Sun
 */

// Gate display symbols
const GATE_SYMBOLS = {
  'H': 'H',
  'X': 'X',
  'Y': 'Y',
  'Z': 'Z',
  'S': 'S',
  'Sdg': 'S†',
  'T': 'T',
  'Tdg': 'T†',
  'SX': '√X',
  'I': 'I',
  'CNOT': 'X',
  'CX': 'X',
  'CZ': 'Z',
  'SWAP': '×',
  'MCM': 'M',
  'MEASURE': 'M',
  'Rx': 'Rx',
  'Ry': 'Ry',
  'Rz': 'Rz',
  'P': 'P',
  'Ph': 'Ph',
  'U': 'U',
  '0': '|0>',
  '1': '|1>',
};

/**
 * ASCII Circuit Drawer.
 *
 * Draws quantum circuits as ASCII art for terminal display.
 *
 * @example
 * const drawer = new ASCIICircuitDrawer(circuit);
 * console.log(drawer.draw());
 */
export class ASCIICircuitDrawer {
  /**
   * @param {Circuit} circuit - The circuit to draw
   */
  constructor(circuit) {
    this.circuit = circuit;
    this.numQubits = circuit.size;
  }

  /**
   * Generate ASCII representation of the circuit.
   * @param {Object} options
   * @param {boolean} options.showAngles - Whether to display angle parameters
   * @returns {string}
   */
  draw(options = {}) {
    const { showAngles = true } = options;

    if (this.circuit.circuit.length === 0) {
      return this._emptyCircuit();
    }

    const layers = this._buildLayers();
    const lines = {};
    const connectorLines = {};

    for (let q = 0; q < this.numQubits; q++) {
      lines[q] = [];
      connectorLines[q] = [];
    }

    for (const layer of layers) {
      const layerWidth = this._getLayerWidth(layer, showAngles);
      this._drawLayer(layer, lines, connectorLines, layerWidth, showAngles);
    }

    return this._assembleOutput(lines, connectorLines);
  }

  _emptyCircuit() {
    const output = [];
    for (let q = 0; q < this.numQubits; q++) {
      output.push(`q${q}: ${'─'.repeat(20)}`);
    }
    return output.join('\n');
  }

  _buildLayers() {
    const layers = [];
    const qubitDepth = new Array(this.numQubits).fill(0);

    for (const gate of this.circuit.circuit) {
      if (gate.name === 'BARRIER') {
        const maxDepth = Math.max(...qubitDepth);
        qubitDepth.fill(maxDepth);
        continue;
      }

      const target = gate.target;
      const controls = gate.controls || [];
      const involved = [target, ...controls];

      const layerIdx = Math.max(...involved.map(q => qubitDepth[q]));

      while (layers.length <= layerIdx) {
        layers.push([]);
      }

      layers[layerIdx].push(gate);

      for (const q of involved) {
        qubitDepth[q] = layerIdx + 1;
      }
    }

    return layers;
  }

  _getLayerWidth(layer, showAngles) {
    let maxWidth = 3;

    for (const gate of layer) {
      const symbol = GATE_SYMBOLS[gate.name] || gate.name.slice(0, 3);
      let width = symbol.length + 2;

      if (showAngles && gate.angles !== undefined) {
        const angleStr = Array.isArray(gate.angles)
          ? gate.angles.map(a => a.toFixed(2)).join(',')
          : gate.angles.toFixed(2);
        width = Math.max(width, symbol.length + angleStr.length + 4);
      }

      maxWidth = Math.max(maxWidth, width);
    }

    return maxWidth;
  }

  _drawLayer(layer, lines, connectorLines, width, showAngles) {
    const qubitContent = {};
    const connections = [];

    for (let q = 0; q < this.numQubits; q++) {
      qubitContent[q] = null;
    }

    for (const gate of layer) {
      const target = gate.target;
      const controls = gate.controls || [];
      const name = gate.name;

      let symbol = GATE_SYMBOLS[name] || name.slice(0, 3);

      if (showAngles && gate.angles !== undefined) {
        const angleStr = Array.isArray(gate.angles)
          ? gate.angles.map(a => a.toFixed(1)).join(',')
          : gate.angles.toFixed(2);
        symbol = `${symbol}(${angleStr})`;
      }

      qubitContent[target] = { type: 'gate', symbol };

      for (const c of controls) {
        qubitContent[c] = { type: 'control', symbol: '●' };
      }

      if (controls.length > 0) {
        const allQubits = [target, ...controls];
        connections.push({
          min: Math.min(...allQubits),
          max: Math.max(...allQubits),
          controls,
          target
        });
      }

      // Handle SWAP specially
      if (name === 'SWAP' && controls.length > 0) {
        qubitContent[controls[0]] = { type: 'gate', symbol: '×' };
        qubitContent[target] = { type: 'gate', symbol: '×' };
      }
    }

    // Draw the layer
    for (let q = 0; q < this.numQubits; q++) {
      const content = qubitContent[q];
      let segment;

      if (content === null) {
        segment = '─'.repeat(width);
      } else if (content.type === 'gate') {
        const symbol = content.symbol;
        const padding = width - symbol.length - 2;
        const leftPad = Math.floor(padding / 2);
        const rightPad = padding - leftPad;
        segment = '─'.repeat(leftPad) + `[${symbol}]` + '─'.repeat(rightPad);
      } else if (content.type === 'control') {
        const padding = width - 1;
        const leftPad = Math.floor(padding / 2);
        const rightPad = padding - leftPad;
        segment = '─'.repeat(leftPad) + '●' + '─'.repeat(rightPad);
      }

      lines[q].push(segment);
    }

    // Draw vertical connectors
    for (let q = 0; q < this.numQubits; q++) {
      let connector = ' '.repeat(width);

      for (const conn of connections) {
        if (conn.min < q && q < conn.max) {
          const mid = Math.floor(width / 2);
          connector = ' '.repeat(mid) + '│' + ' '.repeat(width - mid - 1);
          break;
        }
      }

      connectorLines[q].push(connector);
    }
  }

  _assembleOutput(lines, connectorLines) {
    const output = [];
    const labelWidth = `q${this.numQubits - 1}: `.length;

    for (let q = 0; q < this.numQubits; q++) {
      const label = `q${q}: `.padStart(labelWidth);
      const wire = lines[q].join('');
      output.push(`${label}${wire}`);

      if (q < this.numQubits - 1) {
        const connector = connectorLines[q].join('');
        if (connector.includes('│')) {
          output.push(' '.repeat(labelWidth) + connector);
        }
      }
    }

    return output.join('\n');
  }
}

/**
 * Probability Visualizer.
 *
 * Displays probability distributions and measurement results.
 */
export class ProbabilityVisualizer {
  /**
   * @param {Circuit} circuit - The circuit to visualize
   */
  constructor(circuit) {
    this.circuit = circuit;
  }

  /**
   * Print probability distribution to console.
   * @param {Object} options
   * @param {number} options.threshold - Minimum probability to display
   * @param {number} options.maxStates - Maximum states to show
   * @param {string} options.sortBy - 'probability' or 'state'
   */
  printProbabilities(options = {}) {
    const { threshold = 1e-4, maxStates = 32, sortBy = 'probability' } = options;

    if (!this.circuit.state) {
      console.log('Circuit has not been executed. Call execute() first.');
      return;
    }

    const probs = this.circuit.getProbabilities();
    const numQubits = this.circuit.size;

    const stateProbs = [];
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] >= threshold) {
        const state = i.toString(2).padStart(numQubits, '0').split('').reverse().join('');
        stateProbs.push({ state, prob: probs[i] });
      }
    }

    if (sortBy === 'probability') {
      stateProbs.sort((a, b) => b.prob - a.prob);
    } else {
      stateProbs.sort((a, b) => a.state.localeCompare(b.state));
    }

    const toShow = stateProbs.slice(0, maxStates);

    console.log('\n' + '='.repeat(50));
    console.log('Probability Distribution');
    console.log('='.repeat(50));

    const maxProb = Math.max(...toShow.map(x => x.prob));
    const barWidth = 30;

    for (const { state, prob } of toShow) {
      const barLen = maxProb > 0 ? Math.round((prob / maxProb) * barWidth) : 0;
      const bar = '█'.repeat(barLen) + '░'.repeat(barWidth - barLen);
      console.log(`  |${state}>  ${bar}  ${prob.toFixed(4)}`);
    }

    if (stateProbs.length > maxStates) {
      console.log(`  ... and ${stateProbs.length - maxStates} more states`);
    }

    console.log('='.repeat(50) + '\n');
  }

  /**
   * Print measurement counts to console.
   * @param {Object} options
   * @param {number} options.maxStates - Maximum states to show
   */
  printCounts(options = {}) {
    const { maxStates = 32 } = options;

    const counts = this.circuit.getCounts();
    if (!counts) {
      console.log('No measurement results. Call execute() first.');
      return;
    }

    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const sorted = Object.entries(counts)
      .map(([state, count]) => ({ state, count, prob: count / total }))
      .sort((a, b) => b.count - a.count);

    console.log('\n' + '-'.repeat(50));
    console.log('Measurement Counts');
    console.log('-'.repeat(50));

    const toShow = sorted.slice(0, maxStates);

    for (const { state, count, prob } of toShow) {
      const pct = (prob * 100).toFixed(1);
      console.log(`  |${state}>: ${count} (${pct}%)`);
    }

    if (sorted.length > maxStates) {
      console.log(`  ... and ${sorted.length - maxStates} more states`);
    }

    console.log('-'.repeat(50));
  }
}

/**
 * Unified Circuit Visualizer.
 *
 * @example
 * const vis = new CircuitVisualizer(circuit);
 * vis.printCircuit();
 * vis.printProbabilities();
 * vis.summary();
 */
export class CircuitVisualizer {
  /**
   * @param {Circuit} circuit - The circuit to visualize
   */
  constructor(circuit) {
    this.circuit = circuit;
    this._asciiDrawer = new ASCIICircuitDrawer(circuit);
    this._probViz = new ProbabilityVisualizer(circuit);
  }

  /**
   * Print ASCII circuit diagram.
   * @param {Object} options
   * @param {boolean} options.showAngles - Whether to display angles
   */
  printCircuit(options = {}) {
    console.log(this._asciiDrawer.draw(options));
  }

  /**
   * Print probability distribution.
   * @param {Object} options
   */
  printProbabilities(options = {}) {
    this._probViz.printProbabilities(options);
  }

  /**
   * Print measurement counts.
   * @param {Object} options
   */
  printCounts(options = {}) {
    this._probViz.printCounts(options);
  }

  /**
   * Print complete circuit summary.
   * @param {Object} options
   * @param {boolean} options.showCircuit - Show circuit diagram
   * @param {boolean} options.showProbabilities - Show probabilities
   * @param {boolean} options.showCounts - Show counts
   */
  summary(options = {}) {
    const {
      showCircuit = true,
      showProbabilities = true,
      showCounts = true
    } = options;

    console.log('\n' + '='.repeat(60));
    console.log('QUANTUM CIRCUIT SUMMARY');
    console.log('='.repeat(60));

    console.log(`\nQubits: ${this.circuit.size}`);
    console.log(`Gates: ${this.circuit.length}`);
    console.log(`Depth: ${this.circuit.depth()}`);
    console.log(`Gate counts: ${JSON.stringify(this.circuit.gateCount())}`);

    if (showCircuit) {
      console.log('\n' + '-'.repeat(60));
      console.log('CIRCUIT DIAGRAM');
      console.log('-'.repeat(60));
      this.printCircuit();
    }

    if (showProbabilities && this.circuit.state) {
      this.printProbabilities();
    }

    if (showCounts && this.circuit.measurements) {
      this.printCounts();
    }

    console.log('\n' + '='.repeat(60) + '\n');
  }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Print ASCII circuit diagram.
 * @param {Circuit} circuit
 * @param {Object} options
 */
export function printCircuit(circuit, options = {}) {
  const drawer = new ASCIICircuitDrawer(circuit);
  console.log(drawer.draw(options));
}

/**
 * Get ASCII circuit diagram as string.
 * @param {Circuit} circuit
 * @param {Object} options
 * @returns {string}
 */
export function drawCircuit(circuit, options = {}) {
  const drawer = new ASCIICircuitDrawer(circuit);
  return drawer.draw(options);
}

/**
 * Print complete circuit summary.
 * @param {Circuit} circuit
 */
export function circuitSummary(circuit) {
  const vis = new CircuitVisualizer(circuit);
  vis.summary();
}

export default {
  ASCIICircuitDrawer,
  ProbabilityVisualizer,
  CircuitVisualizer,
  printCircuit,
  drawCircuit,
  circuitSummary,
};
