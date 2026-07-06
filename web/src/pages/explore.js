import Head from "next/head";
import dynamic from "next/dynamic";

const ExplorePage = dynamic(() => import("@/explore/ExplorePage"), { ssr: false });

export default function Explore() {
  return (
    <>
      <Head>
        <title>Explore</title>
      </Head>
      <ExplorePage />
    </>
  );
}
