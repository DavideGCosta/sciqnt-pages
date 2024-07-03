// src/theme/ThemeContext.tsx
import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

interface ThemeContextType {
  theme: 'dark' | 'light';
  toggleTheme: () => void;
}

interface ThemeProviderProps {
  children: ReactNode;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const { colorMode, setColorMode } = useColorMode();
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  useEffect(() => {
    if (ExecutionEnvironment.canUseDOM) {
      const storedTheme = localStorage.getItem('theme') as 'dark' | 'light';
      if (storedTheme) {
        setTheme(storedTheme);
        setColorMode(storedTheme);
      } else {
        setTheme(colorMode);
      }
    }
  }, [colorMode, setColorMode]);

  useEffect(() => {
    if (ExecutionEnvironment.canUseDOM) {
      const handleThemeChange = () => {
        const newTheme = localStorage.getItem('theme') as 'dark' | 'light';
        if (newTheme) {
          setColorMode(newTheme);
          setTheme(newTheme);
        }
      };

      const handleMessage = (event: MessageEvent) => {
        console.log("Message received in iframe:", event.data); // Debugging line
        if (event.data.type === 'CHANGE_THEME') {
          const newTheme = event.data.theme as 'dark' | 'light';
          setColorMode(newTheme);
          setTheme(newTheme);
          localStorage.setItem('theme', newTheme);
        }
      };

      window.addEventListener('themeChanged', handleThemeChange);
      window.addEventListener('message', handleMessage);

      return () => {
        window.removeEventListener('themeChanged', handleThemeChange);
        window.removeEventListener('message', handleMessage);
      };
    }
  }, [setColorMode]);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setColorMode(newTheme);
    setTheme(newTheme);
    if (ExecutionEnvironment.canUseDOM) {
      localStorage.setItem('theme', newTheme);
      window.dispatchEvent(new Event('themeChanged'));
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};