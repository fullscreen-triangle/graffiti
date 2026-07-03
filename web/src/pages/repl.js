import Head from "next/head";
import dynamic from "next/dynamic";

const Repl = dynamic(() => import("@/components/Repl"), { ssr: false });

export default function ReplPage() {
  return (
    <>
      <Head>
        <title>REPL</title>
      </Head>
      <Repl />
    </>
  );
}
