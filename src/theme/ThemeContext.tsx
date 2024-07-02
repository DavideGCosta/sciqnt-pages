// src/theme/ThemeContext.tsx
import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}

interface ThemeProviderProps {
  children: ReactNode;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const { colorMode, setColorMode } = useColorMode();
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  useEffect(() => {
    if (ExecutionEnvironment.canUseDOM) {
      const storedTheme = localStorage.getItem('theme') as 'light' | 'dark';
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
        const newTheme = localStorage.getItem('theme') as 'light' | 'dark';
        if (newTheme) {
          setColorMode(newTheme);
          setTheme(newTheme);
        }
      };

      window.addEventListener('themeChanged', handleThemeChange);

      return () => {
        window.removeEventListener('themeChanged', handleThemeChange);
      };
    }
  }, [setColorMode]);

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
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