import React from 'react';
import { VENN_GAP } from '../../constants/venn.js';

export default function VennDefs({ config }) {
  const { width, height, minX = 0, minY = 0, circles } = config;
  const inc = circles.map((circle) => ({ ...circle, rIn: circle.r - VENN_GAP, rOut: circle.r + VENN_GAP }));
  const circleByKey = Object.fromEntries(inc.map((circle) => [circle.key, circle]));

  const clipPaths = [];
  const buildClip = (keys) => {
    if (keys.length === 1) {
      const circle = circleByKey[keys[0]];
      return <circle key={keys.join('')} cx={circle.cx} cy={circle.cy} r={circle.rIn} />;
    }
    const last = circleByKey[keys[keys.length - 1]];
    return (
      <circle
        key={keys.join('')}
        cx={last.cx}
        cy={last.cy}
        r={last.rIn}
        clipPath={`url(#venn-clip-${keys.slice(0, -1).join('')})`}
      />
    );
  };

  for (let mask = 1; mask < (1 << circles.length); mask += 1) {
    const keys = circles.map((circle) => circle.key).filter((_, index) => mask & (1 << index));
    clipPaths.push(
      <clipPath key={`clip-${keys.join('')}`} id={`venn-clip-${keys.join('')}`} clipPathUnits="userSpaceOnUse">
        {buildClip(keys)}
      </clipPath>,
    );
  }

  const masks = [];
  for (let mask = 1; mask < (1 << circles.length); mask += 1) {
    const excluded = circles.map((circle) => circle.key).filter((_, index) => !(mask & (1 << index)));
    if (excluded.length === 0) continue;
    masks.push(
      <mask key={`mask-${excluded.join('')}`} id={`venn-mask-not-${excluded.join('')}`} maskUnits="userSpaceOnUse" x={minX} y={minY} width={width} height={height}>
        <rect x={minX} y={minY} width={width} height={height} fill="white" />
        {excluded.map((key) => {
          const circle = circleByKey[key];
          return <circle key={key} cx={circle.cx} cy={circle.cy} r={circle.rOut} fill="black" />;
        })}
      </mask>,
    );
  }

  return <defs>{clipPaths}{masks}</defs>;
}
