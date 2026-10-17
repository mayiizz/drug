import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params;
  const search = req.nextUrl.search;
  const upstream = `https://rest.uniprot.org/${path.join("/")}${search}`;

  const res = await fetch(upstream, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (compatible; CURA-StructureViewer/1.0; +https://localhost)",
      Accept: "application/json",
    },
    next: { revalidate: 86400 },
  });

  if (!res.ok) {
    return NextResponse.json(
      { error: `UniProt request failed (${res.status})` },
      { status: res.status }
    );
  }

  const data = await res.json();
  return NextResponse.json(data, {
    headers: { "Cache-Control": "public, max-age=86400" },
  });
}
