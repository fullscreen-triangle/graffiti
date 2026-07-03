import { Suspense } from "react";
import dynamic from "next/dynamic";
import Head from "next/head";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";

const SceneModel = dynamic(() => import("@/components/SceneModel"), {
  ssr: false,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Graffiti</title>
        <meta name="description" content="Graffiti" />
      </Head>

      <div className="h-screen w-screen bg-black">
        <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 5, 5]} intensity={0.8} />
          <Suspense fallback={null}>
            <SceneModel />
            <Environment preset="city" />
          </Suspense>
          <OrbitControls enablePan enableZoom enableRotate autoRotate={false} />
        </Canvas>
      </div>
    </>
  );
}
