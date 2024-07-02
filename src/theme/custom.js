import React from 'react';
import {useColorMode} from '@docusaurus/theme-common';

const Example = () => {
  const { setColorMode } = useColorMode();

  useEffect(() => {
    const handleThemeChange = () => {
      const theme = localStorage.getItem('theme') || 'light';
      setColorMode(theme);
    };

    window.addEventListener('themeChange', handleThemeChange);

    // Set initial theme based on localStorage
    handleThemeChange();

    return () => {
      window.removeEventListener('themeChange', handleThemeChange);
    };
  }, [setColorMode]);
}