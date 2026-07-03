import React, { useEffect, useRef } from "react";
import { useGLTF, Center } from "@react-three/drei";

const MODEL_PATH = "/uxrzone_pointcloud_room_free.glb";

export default function SceneModel(props) {
  const { scene } = useGLTF(MODEL_PATH);
  const ref = useRef();

  // Loaded once, never rotated or animated -- a static point cloud room.
  useEffect(() => {
    if (ref.current) {
      ref.current.rotation.set(0, 0, 0);
    }
  }, []);

  return (
    <Center>
      <primitive ref={ref} object={scene} {...props} />
    </Center>
  );
}

useGLTF.preload(MODEL_PATH);
