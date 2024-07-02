// src/theme/Layout/Provider/index.tsx
import React from 'react';
import { composeProviders } from '@docusaurus/theme-common';
import {
  ColorModeProvider,
  AnnouncementBarProvider,
  DocsPreferredVersionContextProvider,
  ScrollControllerProvider,
  NavbarProvider,
  PluginHtmlClassNameProvider,
} from '@docusaurus/theme-common/internal';
import type { Props } from '@theme/Layout/Provider';
import { ThemeProvider } from '../../ThemeContext';

const Provider = composeProviders([
  ColorModeProvider, // Ensure this is the first provider
  AnnouncementBarProvider,
  ScrollControllerProvider,
  DocsPreferredVersionContextProvider,
  PluginHtmlClassNameProvider,
  NavbarProvider,
]);

export default function LayoutProvider({ children }: Props): JSX.Element {
  return (
    <Provider>
      <ThemeProvider>
        {children}
      </ThemeProvider>
    </Provider>
  );
}