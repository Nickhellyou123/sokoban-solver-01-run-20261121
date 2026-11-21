// Sokoban solver: push-optimal, room by room.
//
// Idea: a box pushed out of a room through its single door lands in the
// hallway, where it can never be pushed back (the player cannot get to the
// far side of the door). So every box stays in its own room, each room is an
// independent puzzle, and the level's push-optimum is the sum of the rooms'.
// Each room is solved optimally with a breadth-first search over pushes:
// a state is the boxes' squares plus the region the player can reach, the
// search expands states in order of pushes made, so the first solved state
// found is push-optimal for that room. Dead squares (no pull-back path from
// any goal) and frozen 2x2 blocks are pruned; both prunings are sound, so
// optimality is preserved. The search budget is a count of stored states,
// never the clock, so two runs produce identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// One search may keep at most this many states per room. The generator only
// keeps rooms its own push search solves well inside this bound, so hitting
// it means the level is not of the published shape; the caller then falls
// back to a whole-level search and, failing that, to an empty answer.
const MAX_STATES = 400_000;

const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outer frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Squares from which a lone box can be pulled back onto some goal: start at
// the goals and walk backwards. A box on any other square can never reach a
// goal, whatever the other boxes do.
function liveSquares(w: number, h: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const size = w * h;
  const live = new Uint8Array(size);
  const queue = goals.slice();
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// A flood fill over open floor (walls and the squares in `occ` block it).
// Returns the lowest square of the reached region; the reached set itself is
// readable as `seen[q] === stamp` until the next call. Each factory instance
// keeps its own stamp array, so two fills can be interleaved.
function makeFlood(w: number, h: number, wall: Uint8Array, occ: Uint8Array): {
  run: (from: number) => { low: number; stamp: number };
  seen: Int32Array;
} {
  const size = w * h;
  const seen = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const run = (from: number): { low: number; stamp: number } => {
    stamp++;
    let low = from;
    let n = 0;
    stack[n++] = from;
    seen[from] = stamp;
    while (n > 0) {
      const p = stack[--n]!;
      if (p < low) low = p;
      const up = p - w;
      const dn = p + w;
      const lf = p - 1;
      const rt = p + 1;
      if (!wall[up] && !occ[up] && seen[up] !== stamp) { seen[up] = stamp; stack[n++] = up; }
      if (!wall[dn] && !occ[dn] && seen[dn] !== stamp) { seen[dn] = stamp; stack[n++] = dn; }
      if (!wall[lf] && !occ[lf] && seen[lf] !== stamp) { seen[lf] = stamp; stack[n++] = lf; }
      if (!wall[rt] && !occ[rt] && seen[rt] !== stamp) { seen[rt] = stamp; stack[n++] = rt; }
    }
    return { low, stamp };
  };
  return { run, seen };
}

// Whether the box just pushed onto `t` sits in a frozen 2x2 block: four
// squares of walls and boxes holding a box off its goal. No box in such a
// block can ever move again, so the position is dead.
function frozenBlock(w: number, wall: Uint8Array, occ: Uint8Array, isGoal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !isGoal[a]) || (occ[b] && !isGoal[b]) || (occ[c] && !isGoal[c]) || (occ[d] && !isGoal[d])) return true;
  }
  return false;
}

// The shortest walk from `from` to `to` around walls and boxes, as lower-case
// moves. Neighbours are tried u, d, l, r, so ties break the same way every run.
function walkTo(w: number, h: number, wall: Uint8Array, occ: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const size = w * h;
  const prev = new Int32Array(size).fill(-1);
  const how: string[] = new Array<string>(size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, ch] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = ch;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

// Push-optimal solution of one (sub-)level, or null when the state budget
// runs out or no solution exists. Breadth-first over pushes: states are
// visited in order of pushes made, so the first solved state is optimal.
function solvePushes(
  w: number,
  h: number,
  wall: Uint8Array,
  goals: number[],
  boxes0: number[],
  player0: number,
  maxStates: number,
): { pushes: number; moves: string } | null {
  const size = w * h;
  const isGoal = new Uint8Array(size);
  for (const g of goals) isGoal[g] = 1;
  const live = liveSquares(w, h, wall, goals);
  for (const b of boxes0) if (!live[b]) return null;
  const solved = (bs: number[]): boolean => bs.every((b) => isGoal[b]);
  if (solved(boxes0)) return { pushes: 0, moves: '' };

  const occ = new Uint8Array(size);
  const reach = makeFlood(w, h, wall, occ);
  const keyFlood = makeFlood(w, h, wall, occ);
  const keyOf = (low: number, bs: number[]): string => low + ':' + bs.join(',');

  // States in discovery order: boxes, player square after the last push,
  // parent state, and the push direction that made the state.
  const boxStates: number[][] = [boxes0.slice()];
  const playerStates: number[] = [player0];
  const parent: number[] = [-1];
  const pushDir: number[] = [-1];
  const visited = new Map<string, number>();
  for (const b of boxes0) occ[b] = 1;
  visited.set(keyOf(keyFlood.run(player0).low, boxes0), 0);
  for (const b of boxes0) occ[b] = 0;

  const reconstruct = (found: number): { pushes: number; moves: string } => {
    occ.fill(0);
    const chain: number[] = [];
    for (let s = found; s > 0; s = parent[s]!) chain.push(s);
    chain.reverse();
    let moves = '';
    let player = player0;
    let cur = boxes0.slice();
    for (const s of chain) {
      const next = boxStates[s]!;
      let from = -1;
      let to = -1;
      for (const b of cur) if (!next.includes(b)) from = b;
      for (const b of next) if (!cur.includes(b)) to = b;
      const d = to - from;
      const dir = DIRS[pushDir[s]!]!;
      for (const b of cur) occ[b] = 1;
      moves += walkTo(w, h, wall, occ, player, from - d);
      for (const b of cur) occ[b] = 0;
      moves += dir[3];
      player = from;
      cur = next.slice();
    }
    return { pushes: chain.length, moves };
  };

  for (let head = 0; head < boxStates.length; head++) {
    const bs = boxStates[head]!;
    const pl = playerStates[head]!;
    for (const b of bs) occ[b] = 1;
    const region = reach.run(pl);
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const behind = b - d;
        const to = b + d;
        if (reach.seen[behind] !== region.stamp) continue;
        if (wall[to] || occ[to] || !live[to]) continue;
        occ[b] = 0;
        occ[to] = 1;
        if (frozenBlock(w, wall, occ, isGoal, to)) {
          occ[to] = 0;
          occ[b] = 1;
          continue;
        }
        const next = bs.slice();
        next[bi] = to;
        next.sort((x, y) => x - y);
        const low = keyFlood.run(b).low;
        const key = keyOf(low, next);
        if (visited.has(key)) {
          occ[to] = 0;
          occ[b] = 1;
          continue;
        }
        if (boxStates.length >= maxStates) {
          for (const x of bs) occ[x] = 0;
          return null;
        }
        visited.set(key, boxStates.length);
        boxStates.push(next);
        playerStates.push(b);
        parent.push(head);
        pushDir.push(di);
        if (solved(next)) {
          for (const x of bs) occ[x] = 0;
          return reconstruct(boxStates.length - 1);
        }
        occ[to] = 0;
        occ[b] = 1;
      }
    }
    for (const x of bs) occ[x] = 0;
  }
  return null;
}

interface Room {
  boxes: number[];
  goals: number[];
}

// Split a level into rooms: cut the dead squares of the player's row (the
// hallway: a box there reaches no goal) out of the floor, and each remaining
// connected part holding boxes is a room. Null when the level is no set of
// rooms (a box in the hallway, a goal cut off, or a part whose boxes and
// goals do not match).
function findRooms(level: Level): Room[] | null {
  const { w, h, wall, goals, boxes, player } = level;
  const size = w * h;
  const live = liveSquares(w, h, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const id = rooms.length;
    const squares = [s];
    part[s] = id;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = id;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    rooms[part[b]]!.boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Play `moves` on the whole level from (player, boxes); returns the final
// player square and box squares. Used only to carry the player between rooms.
function playMoves(level: Level, player: number, boxes: number[], moves: string): { player: number; boxes: number[] } {
  const w = level.w;
  const boxSet = new Set(boxes);
  let p = player;
  for (const ch of moves.toLowerCase()) {
    const d = ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
    const next = p + d;
    if (boxSet.has(next)) {
      boxSet.delete(next);
      boxSet.add(next + d);
    }
    p = next;
  }
  return { player: p, boxes: [...boxSet].sort((a, b) => a - b) };
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = findRooms(level);
  if (rooms === null) {
    const whole = solvePushes(level.w, level.h, level.wall, level.goals, level.boxes, level.player, MAX_STATES);
    return whole ? whole.moves : '';
  }
  let moves = '';
  let player = level.player;
  // Each room's boxes as they stand now; a finished room's boxes sit on its
  // goals, so they count as walls for the rooms still to come.
  const current: number[][] = rooms.map((r) => r.boxes.slice());
  for (let k = 0; k < rooms.length; k++) {
    const room = rooms[k]!;
    const subWall = level.wall.slice();
    for (let j = 0; j < rooms.length; j++) {
      if (j === k) continue;
      const src = j < k ? rooms[j]!.goals : current[j]!;
      for (const b of src) subWall[b] = 1;
    }
    const res = solvePushes(level.w, level.h, subWall, room.goals, current[k]!, player, MAX_STATES);
    if (res === null) {
      const whole = solvePushes(level.w, level.h, level.wall, level.goals, level.boxes, level.player, MAX_STATES);
      return whole ? whole.moves : '';
    }
    moves += res.moves;
    const after = playMoves(level, player, rooms.flatMap((r, j) => (j < k ? r.goals : current[j]!)), res.moves);
    player = after.player;
    current[k] = room.goals.slice();
  }
  return moves;
}

function solve(text: string): string {
  try {
    return solveLevel(text);
  } catch {
    // Never leave an instance unanswered: an empty answer is valid, worth 0.
    return '';
  }
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
