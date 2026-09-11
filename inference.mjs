export function infer(input, weights, layers) {
  if (input.length !== 784) throw new Error('Expected 784 input pixels');
  let values = input;
  layers.forEach((layer,index) => {
    const next = new Float32Array(layer.output);
    const biasOffset = layer.offset + layer.input * layer.output;
    for (let j=0; j<layer.output; j++) {
      let sum = weights[biasOffset+j];
      const start = layer.offset + j*layer.input;
      for(let i=0; i<layer.input; i++) sum += values[i]*weights[start+i];
      next[j] = index < layers.length-1 ? Math.max(0,sum) : sum;
    }
    values=next;
  });
  const maximum=Math.max(...values);
  const exps=Array.from(values,v=>Math.exp(v-maximum));
  const total=exps.reduce((a,b)=>a+b,0);
  return exps.map(v=>v/total);
}
