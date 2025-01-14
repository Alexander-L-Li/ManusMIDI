"use client";
import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="h-screen w-screen bg-gradient-to-tr from-lime-100 to-emerald-100">
      <div className="flex flex-col justify-center items-center h-full">
        <div className="lg:text-8xl md:text-6xl text-4xl font-bold text-center text-black lg:pb-5 md:pb-3 pb-1">
          ManusMIDI
        </div>
        <div className="lg:text-5xl md:text-4xl text-xl font-light text-center text-black">
          {" "}
          Create music — with just your hands!
        </div>
        <Link href={`../twiggy`}>
          <button
            className="transition duration-300 ease-in-out rounded-full border-black lg:border-4 md:border-4 border-2 bg-clear lg:p-8 md:p-5 p-2 lg:m-16 md:m-10 m-5 lg:text-4xl md:text-2xl text-xl font-medium hover:bg-green-300 hover:drop-shadow-lg text-black"
            onclick="/main"
          >
            Give it a try!
          </button>
        </Link>
      </div>
    </div>
  );
}
