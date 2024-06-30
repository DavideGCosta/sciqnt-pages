import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/sciqnt-pages/__docusaurus/debug',
    component: ComponentCreator('/sciqnt-pages/__docusaurus/debug', '7b6'),
    exact: true
  },
  {
    path: '/sciqnt-pages/__docusaurus/debug/config',
    component: ComponentCreator('/sciqnt-pages/__docusaurus/debug/config', 'a2c'),
    exact: true
  },
  {
    path: '/sciqnt-pages/__docusaurus/debug/content',
    component: ComponentCreator('/sciqnt-pages/__docusaurus/debug/content', '083'),
    exact: true
  },
  {
    path: '/sciqnt-pages/__docusaurus/debug/globalData',
    component: ComponentCreator('/sciqnt-pages/__docusaurus/debug/globalData', '000'),
    exact: true
  },
  {
    path: '/sciqnt-pages/__docusaurus/debug/metadata',
    component: ComponentCreator('/sciqnt-pages/__docusaurus/debug/metadata', '1ac'),
    exact: true
  },
  {
    path: '/sciqnt-pages/__docusaurus/debug/registry',
    component: ComponentCreator('/sciqnt-pages/__docusaurus/debug/registry', 'fd2'),
    exact: true
  },
  {
    path: '/sciqnt-pages/__docusaurus/debug/routes',
    component: ComponentCreator('/sciqnt-pages/__docusaurus/debug/routes', '3cf'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog',
    component: ComponentCreator('/sciqnt-pages/blog', '782'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/archive',
    component: ComponentCreator('/sciqnt-pages/blog/archive', '512'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/first-blog-post',
    component: ComponentCreator('/sciqnt-pages/blog/first-blog-post', '904'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/long-blog-post',
    component: ComponentCreator('/sciqnt-pages/blog/long-blog-post', '09b'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/mdx-blog-post',
    component: ComponentCreator('/sciqnt-pages/blog/mdx-blog-post', '1a4'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/tags',
    component: ComponentCreator('/sciqnt-pages/blog/tags', '033'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/tags/docusaurus',
    component: ComponentCreator('/sciqnt-pages/blog/tags/docusaurus', '0f5'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/tags/facebook',
    component: ComponentCreator('/sciqnt-pages/blog/tags/facebook', '537'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/tags/hello',
    component: ComponentCreator('/sciqnt-pages/blog/tags/hello', 'd52'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/tags/hola',
    component: ComponentCreator('/sciqnt-pages/blog/tags/hola', '98e'),
    exact: true
  },
  {
    path: '/sciqnt-pages/blog/welcome',
    component: ComponentCreator('/sciqnt-pages/blog/welcome', 'eb2'),
    exact: true
  },
  {
    path: '/sciqnt-pages/docs',
    component: ComponentCreator('/sciqnt-pages/docs', '02c'),
    routes: [
      {
        path: '/sciqnt-pages/docs',
        component: ComponentCreator('/sciqnt-pages/docs', '768'),
        routes: [
          {
            path: '/sciqnt-pages/docs',
            component: ComponentCreator('/sciqnt-pages/docs', 'e3f'),
            routes: [
              {
                path: '/sciqnt-pages/docs/api-reference/create-a-blog-post',
                component: ComponentCreator('/sciqnt-pages/docs/api-reference/create-a-blog-post', '358'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/api-reference/create-a-document',
                component: ComponentCreator('/sciqnt-pages/docs/api-reference/create-a-document', '8ab'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/api-reference/create-a-page',
                component: ComponentCreator('/sciqnt-pages/docs/api-reference/create-a-page', '6bc'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/api-reference/deploy-your-site',
                component: ComponentCreator('/sciqnt-pages/docs/api-reference/deploy-your-site', 'a0d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/category/api-reference',
                component: ComponentCreator('/sciqnt-pages/docs/category/api-reference', 'd81'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/category/financial-news-strategy',
                component: ComponentCreator('/sciqnt-pages/docs/category/financial-news-strategy', 'df0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/category/strategies',
                component: ComponentCreator('/sciqnt-pages/docs/category/strategies', '963'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/category/strategy-2',
                component: ComponentCreator('/sciqnt-pages/docs/category/strategy-2', 'b59'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/intro',
                component: ComponentCreator('/sciqnt-pages/docs/intro', 'c13'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/strategies/financial-news/clustering',
                component: ComponentCreator('/sciqnt-pages/docs/strategies/financial-news/clustering', '0ee'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/strategies/financial-news/sentence-embeddings',
                component: ComponentCreator('/sciqnt-pages/docs/strategies/financial-news/sentence-embeddings', '303'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/strategies/financial-news/sourcing-news',
                component: ComponentCreator('/sciqnt-pages/docs/strategies/financial-news/sourcing-news', '58b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/strategies/strategy2/Clustering',
                component: ComponentCreator('/sciqnt-pages/docs/strategies/strategy2/Clustering', '03e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/strategies/strategy2/Sentence Embeddings',
                component: ComponentCreator('/sciqnt-pages/docs/strategies/strategy2/Sentence Embeddings', '4db'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/sciqnt-pages/docs/strategies/strategy2/Sourcing News',
                component: ComponentCreator('/sciqnt-pages/docs/strategies/strategy2/Sourcing News', 'ab6'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/sciqnt-pages/',
    component: ComponentCreator('/sciqnt-pages/', '232'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
