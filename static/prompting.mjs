// Reuse a visual prompt for at most two additional 2× doublings.
export const maxExtraDepth = 2;

const nearby = (point, center, width, fraction) =>
  point.every((value, axis) => Math.abs(value - center[axis]) <= width * fraction);

const cropWidth = ({bounds}) => bounds[2] - bounds[0];

function sameFocus(first, second) {
  return first.zoomFactor === second.zoomFactor && nearby(second.center, first.center, cropWidth(first), 0.10);
}

export function promptApplies(analysis, target) {
  if (analysis.settings !== target.settings) return false;
  const extraDepth = target.depth + Math.log2(target.zoomFactor)
    - analysis.depth - Math.log2(analysis.zoomFactor);
  if (extraDepth < 0 || extraDepth > maxExtraDepth) return false;
  if (analysis.frameId === target.frameId) return sameFocus(analysis, target);

  let point = target.center;
  for (const step of target.trail) {
    const [left, top] = step.bounds;
    const width = cropWidth(step);
    point = [left + point[0] * width, top + point[1] * width];
    if (step.frameId === analysis.frameId) {
      return sameFocus(analysis, step) && nearby(point, analysis.center, cropWidth(analysis), 0.25);
    }
  }
  return false;
}
