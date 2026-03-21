const portArg = Number(Deno.args[0] ?? "8000");
const port = Number.isFinite(portArg) && portArg > 0 ? Math.floor(portArg) : 8000;
const host = "127.0.0.1";
const root = Deno.cwd();

function contentType(path: string): string {
  if (path.endsWith(".html")) return "text/html; charset=utf-8";
  if (path.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (path.endsWith(".css")) return "text/css; charset=utf-8";
  if (path.endsWith(".json")) return "application/json; charset=utf-8";
  if (path.endsWith(".wasm")) return "application/wasm";
  if (path.endsWith(".simd")) return "text/plain; charset=utf-8";
  return "application/octet-stream";
}

function resolveFsPath(urlPath: string): string | null {
  const decoded = decodeURIComponent(urlPath);
  const route = decoded === "/" ? "/docs/demo_explorer.html" : decoded;
  const parts = route
    .split("/")
    .filter((part) => part.length > 0 && part !== ".");
  if (parts.some((part) => part === "..")) {
    return null;
  }
  return `${root}/${parts.join("/")}`;
}

async function servePath(fsPath: string): Promise<Response> {
  let candidate = fsPath;
  try {
    const stat = await Deno.stat(candidate);
    if (stat.isDirectory) {
      candidate = `${candidate}/index.html`;
    }
  } catch {
    return new Response("not found", { status: 404 });
  }

  try {
    const body = await Deno.readFile(candidate);
    return new Response(body, {
      status: 200,
      headers: {
        "content-type": contentType(candidate),
        "cache-control": "no-store",
      },
    });
  } catch {
    return new Response("not found", { status: 404 });
  }
}

console.log(`serving ${root} on http://${host}:${port}/`);

Deno.serve({ hostname: host, port }, (request) => {
  const fsPath = resolveFsPath(new URL(request.url).pathname);
  if (!fsPath) {
    return new Response("forbidden", { status: 403 });
  }
  return servePath(fsPath);
});
