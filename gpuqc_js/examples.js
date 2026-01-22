/**
 * Example usage of the gpuqc_js quantum circuit simulator.
 *
 * This module demonstrates:
 * 1. Creating quantum circuits
 * 2. Building common quantum states (Bell state, GHZ state)
 * 3. Using rotation gates
 * 4. Mid-circuit measurements
 * 5. Analyzing results
 * 6. Circuit visualization
 *
 * Run with: node examples.js
 * Or: npm run examples
 *
 * @author Nathaniel Sun
 */

import { Circuit } from './circuit.js';
import {
  CircuitVisualizer,
  printCircuit,
  circuitSummary
} from './visualization.js';
import {
  getAvailableDevices,
  getCurrentBackend,
  printStatevector,
  printMemoryStats,
  countsToProbabilities,
} from './utils.js';

/**
 * Run all example circuits.
 */
export async function runExamples() {
  // Check available devices
  console.log('='.repeat(60));
  console.log('Available Backends:');
  console.log('='.repeat(60));

  const devices = await getAvailableDevices();
  for (const [device, available] of Object.entries(devices)) {
    const status = available ? 'available' : 'not available';
    console.log(`  ${device}: ${status}`);
  }
  console.log();

  // -----------------------------------------------------
  // Example 1: Bell State
  // -----------------------------------------------------
  console.log('='.repeat(60));
  console.log('Example 1: Bell State');
  console.log('='.repeat(60));

  const bell = new Circuit(2, { device: 'cpu' });
  bell.h(0).cnot(0, 1);

  console.log('\nCircuit Diagram (ASCII):');
  console.log('-'.repeat(40));
  printCircuit(bell);
  console.log('-'.repeat(40));

  await bell.execute(1000);

  console.log('\nMeasurement counts:');
  const bellCounts = bell.getCounts();
  for (const [state, count] of Object.entries(bellCounts).sort()) {
    console.log(`  |${state}>: ${count}`);
  }

  console.log('\nProbabilities:');
  const bellProbs = countsToProbabilities(bellCounts);
  for (const [state, prob] of Object.entries(bellProbs).sort()) {
    console.log(`  |${state}>: ${prob.toFixed(3)}`);
  }

  console.log('\nStatevector (before measurement collapse):');
  bell._runCircuit();
  printStatevector(bell.getStatevector());

  // -----------------------------------------------------
  // Example 2: GHZ State (3 qubits)
  // -----------------------------------------------------
  console.log('\n' + '='.repeat(60));
  console.log('Example 2: GHZ State (3 qubits)');
  console.log('='.repeat(60));

  const ghz = new Circuit(3, { device: 'cpu' });
  ghz.h(0).cnot(0, 1).cnot(1, 2);

  console.log('\nCircuit Diagram (ASCII):');
  console.log('-'.repeat(40));
  printCircuit(ghz);
  console.log('-'.repeat(40));

  await ghz.execute(1000);

  console.log('\nMeasurement counts:');
  for (const [state, count] of Object.entries(ghz.getCounts()).sort()) {
    console.log(`  |${state}>: ${count}`);
  }

  // -----------------------------------------------------
  // Example 3: Rotation Gates
  // -----------------------------------------------------
  console.log('\n' + '='.repeat(60));
  console.log('Example 3: Rotation Gates');
  console.log('='.repeat(60));

  const rot = new Circuit(1, { device: 'cpu' });
  rot.ry(0, Math.PI / 4);

  console.log('\nCircuit Diagram (ASCII):');
  console.log('-'.repeat(40));
  printCircuit(rot, { showAngles: true });
  console.log('-'.repeat(40));

  await rot.execute(1000);

  console.log('\nRy(π/4) on |0>:');
  console.log('Expected: ~85% |0>, ~15% |1>');
  console.log('Measured:');
  for (const [state, count] of Object.entries(rot.getCounts()).sort()) {
    console.log(`  |${state}>: ${count} (${(count / 10).toFixed(1)}%)`);
  }

  // -----------------------------------------------------
  // Example 4: Quantum Teleportation Circuit
  // -----------------------------------------------------
  console.log('\n' + '='.repeat(60));
  console.log('Example 4: Quantum Teleportation Setup');
  console.log('='.repeat(60));

  const teleport = new Circuit(3, { device: 'cpu' });
  teleport.rx(0, Math.PI / 3);
  teleport.h(1);
  teleport.cnot(1, 2);
  teleport.cnot(0, 1);
  teleport.h(0);

  console.log('\nCircuit Diagram (ASCII):');
  console.log('-'.repeat(40));
  printCircuit(teleport, { showAngles: true });
  console.log('-'.repeat(40));

  console.log(`\nCircuit depth: ${teleport.depth()}`);
  console.log(`Gate counts: ${JSON.stringify(teleport.gateCount())}`);

  await teleport.execute(1000);
  console.log('\nMeasurement distribution:');
  for (const [state, count] of Object.entries(teleport.getCounts()).sort()) {
    console.log(`  |${state}>: ${count}`);
  }

  // -----------------------------------------------------
  // Example 5: Superposition of all states
  // -----------------------------------------------------
  console.log('\n' + '='.repeat(60));
  console.log('Example 5: Uniform Superposition (4 qubits)');
  console.log('='.repeat(60));

  const uniform = new Circuit(4, { device: 'cpu' });
  for (let i = 0; i < 4; i++) {
    uniform.h(i);
  }

  console.log('\nCircuit Diagram (ASCII):');
  console.log('-'.repeat(40));
  printCircuit(uniform);
  console.log('-'.repeat(40));

  await uniform.execute(4000);

  console.log('\nAll 16 states should have ~250 counts each:');
  const uniformCounts = uniform.getCounts();
  for (const [state, count] of Object.entries(uniformCounts).sort()) {
    const bar = '#'.repeat(Math.floor(count / 25));
    console.log(`  |${state}>: ${count.toString().padStart(4)} ${bar}`);
  }

  // -----------------------------------------------------
  // Example 6: Method Chaining
  // -----------------------------------------------------
  console.log('\n' + '='.repeat(60));
  console.log('Example 6: Fluent API (Method Chaining)');
  console.log('='.repeat(60));

  const result = new Circuit(2, { device: 'cpu' })
    .h(0)
    .cnot(0, 1)
    .z(1);

  await result.execute(500);

  console.log('\nCircuit Diagram (ASCII):');
  console.log('-'.repeat(40));
  printCircuit(result);
  console.log('-'.repeat(40));

  console.log('\nResults:', result.getCounts());

  // -----------------------------------------------------
  // Example 7: Full Circuit Summary
  // -----------------------------------------------------
  console.log('\n' + '='.repeat(60));
  console.log('Example 7: Complete Circuit Summary');
  console.log('='.repeat(60));

  const complex = new Circuit(3, { device: 'cpu' });
  complex.h(0);
  complex.h(1);
  complex.cnot(0, 2);
  complex.cnot(1, 2);
  complex.t(2);
  complex.s(0);

  await complex.execute(1000);

  const vis = new CircuitVisualizer(complex);
  vis.summary();

  // -----------------------------------------------------
  // Memory stats
  // -----------------------------------------------------
  console.log('\n' + '='.repeat(60));
  console.log('Memory Statistics');
  console.log('='.repeat(60));
  printMemoryStats();
  console.log(`Current backend: ${getCurrentBackend()}`);

  // Cleanup
  bell.dispose();
  ghz.dispose();
  rot.dispose();
  teleport.dispose();
  uniform.dispose();
  result.dispose();
  complex.dispose();

  console.log('\n' + '='.repeat(60));
  console.log('All examples completed successfully!');
  console.log('='.repeat(60));
}

// Run if executed directly
runExamples().catch(console.error);

export default runExamples;
