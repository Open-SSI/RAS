// Score inputs with a delivered-model source in one Deno process: deno run --allow-read score.js MODEL INPUTS OUT
// MODEL defines predict(input); INPUTS is a JSON list of strings; OUT receives the JSON list of scores.
const [modelPath, inputsPath, outPath] = Deno.args;
const predict = new Function(Deno.readTextFileSync(modelPath) + "\nreturn predict;")();
const inputs = JSON.parse(Deno.readTextFileSync(inputsPath));
Deno.writeTextFileSync(outPath, JSON.stringify(inputs.map((x) => Number(predict(x)))));
