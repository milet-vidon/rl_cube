const test = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

// Exercise the page's actual click handlers against the JSON request consumed by DemoUI.
function element() {
  return {
    children: [],
    append(...children) { this.children.push(...children); },
    setAttribute(name, value) { this[name] = value; },
    querySelectorAll() { return []; }
  };
}

test('face-axis buttons submit exclusive wrist motion and are disabled in replay', async () => {
  const page = readFileSync(join(__dirname, 'natural_face90_demo.html'), 'utf8');
  const ids = Object.fromEntries(Array.from(page.matchAll(/ id="([^"]+)"/g), m => [m[1], element()]));
  const actions = [];
  const context = vm.createContext({
    document: {
      getElementById: id => ids[id],
      createElement: element,
      addEventListener() {}
    },
    draw() {},
    async fetch(path, init) {
      assert.equal(path, '/action');
      actions.push(JSON.parse(init.body).action);
      return {ok: true};
    }
  });
  const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
  vm.runInContext(script, context);
  const [minus, plus] = ids.faceOrbit.children[0].children[1].children;
  ids.ticks.onclick({target: {dataset: {ticks: '5'}}});
  function frame(phase, revision) {
    context.frame = {phase, revision, session: 'test'};
    vm.runInContext('connected=true; latest=frame; sync()', context);
  }
  frame('replay', 1);
  assert.ok(minus.disabled && plus.disabled);
  await plus.onclick();
  assert.deepEqual(actions, []);
  const hold = {move:[0,0,0], rotate:[0,0,0], orbit:0, thumb:0, index:0, opposition:0, ticks:5};
  for (const [revision, button, orbit] of [[2, minus, -1], [3, plus, 1], [4, ids.hold, 0]]) {
    frame('manual', revision);
    assert.equal(button.disabled, false);
    await button.onclick();
    assert.deepEqual(actions.at(-1), {...hold, orbit});
    assert.equal(button.disabled, true);
  }
  assert.equal(actions.length, 3);
});
