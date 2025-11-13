const fs = require("fs");

const formatInTimeZone = (date, timeZone) => {
  const fmt = new Intl.DateTimeFormat("en-GB", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZoneName: "short",
  });
  const parts = fmt.formatToParts(date);
  const p = {};
  for (const part of parts) p[part.type] = part.value;
  return `${p.year}-${p.month}-${p.day}T${p.hour}:${p.minute}:${p.second} ${p.timeZoneName}`;
};

const out = { time: formatInTimeZone(new Date(), "Europe/Vienna") };
fs.writeFileSync("resultTime.json", JSON.stringify(out, null, 2));
console.log("Wrote resultTime.json:", out.time);
