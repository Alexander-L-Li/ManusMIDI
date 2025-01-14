"use client";
import Image from "next/image";
import { useEffect, useRef } from 'react';
import VideoStream from '../../components/VideoStream';

export default function Twiggy() {
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.src = 'http://localhost:5000/video_feed';
    }
  }, []);
  return (
    <div className="h-screen w-screen bg-gradient-to-tr from-lime-100 to-emerald-100">
      <div className="flex flex-col items-center h-full">
        <div className = "p-10 lg:text-6xl md:text-5xl text-5xl font-semibold text-center lg:pb-5 md:pb-3 pb-1 text-black">Twiggy</div>
          {/* <div className = "h-full w-full"> */}
          <div className = "h-min w-10/12 border-black rounded-xl border-4 overflow-hidden mt-10">
          <VideoStream />
          </div>
          {/* </div> */}
      </div>
    </div>
  );
}
