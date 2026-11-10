// Draft B of the hero2 revision: the same figure with the alternative lane glyphs (docs/choice.md).
import { makeDraw, figure as f } from './hero2.js';
export const figure = { ...f, name: 'hero2_b' };
export default makeDraw({ lanes: 'B' });
