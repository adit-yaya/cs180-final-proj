import assert from 'node:assert/strict';
import test from 'node:test';
import {promptApplies} from '../static/prompting.mjs';

const crop = {frameId: 1, depth: 0, center: [.5, .5], zoomFactor: 2,
  bounds: [.25, .25, .75, .75], settings: 'gentle', trail: []};
const child = {...crop, frameId: 2, depth: 1, trail: [crop]};
const grandchild = {...crop, frameId: 3, depth: 2, trail: [child, crop]};

test('a prepared prompt applies to the same aimed crop', () => {
  assert.equal(promptApplies(crop, crop), true);
  assert.equal(promptApplies(crop, {...crop, center: [.53, .52]}), true);
  assert.equal(promptApplies(crop, {...crop, center: [.61, .52]}), false);
  assert.equal(promptApplies(crop, {...crop, zoomFactor: 4}), false);
});

test('reuse stays near the original click for two additional doublings', () => {
  assert.equal(promptApplies(crop, child), true);
  assert.equal(promptApplies(crop, grandchild), true);
  assert.equal(promptApplies(crop, {...child, center: [.98, .5]}), false);
  assert.equal(promptApplies(crop, {...grandchild, zoomFactor: 4}), false);
});

test('an edge click and an interior click can share crop bounds but target different features', () => {
  const corner = {...crop, center: [.02, .5], bounds: [0, .25, .5, .75]};
  const another = {...corner, center: [.23, .5]};
  assert.equal(promptApplies(corner, another), false);
  assert.equal(promptApplies(corner, {...child, trail: [another]}), false);
});

test('changing settings or opening another image invalidates the old prompt', () => {
  assert.equal(promptApplies(crop, {...child, settings: 'medium'}), false);
  assert.equal(promptApplies(crop, {...crop, frameId: 9}), false);
  assert.equal(promptApplies(crop, {...child, trail: [{...crop, frameId: 7}]}), false);
});
