import Link from "next/link";
import React from "react";

const Navbar = () => {
  return (
    <header className="absolute top-0 left-0 z-10 p-6">
      <Link
        href="/repl"
        className="text-sm font-medium text-light underline underline-offset-4 hover:opacity-70"
      >
        REPL
      </Link>
    </header>
  );
};

export default Navbar;
