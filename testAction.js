const fs = require("fs");

const out = { time: new Date().toISOString() };
fs.writeFileSync("resultTime.json", JSON.stringify(out, null, 2));
console.log("Wrote resultTime.json:", out.time);
