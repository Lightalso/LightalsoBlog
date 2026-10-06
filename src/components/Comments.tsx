import Giscus, { type Theme } from "@giscus/react";
import { GISCUS } from "@/constants";
import { useEffect, useState } from "react";

interface CommentsProps {
  lightTheme?: Theme;
  darkTheme?: Theme;
}

function getCurrentTheme(): "light" | "dark" {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

export default function Comments({
  lightTheme = "light",
  darkTheme = "dark",
}: CommentsProps) {
  const [theme, setTheme] = useState(getCurrentTheme);

  useEffect(() => {
    const observer = new MutationObserver(() => setTheme(getCurrentTheme()));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => observer.disconnect();
  }, []);

  return (
    <div className="mt-8">
      <Giscus {...GISCUS} theme={theme === "light" ? lightTheme : darkTheme} />
    </div>
  );
}
