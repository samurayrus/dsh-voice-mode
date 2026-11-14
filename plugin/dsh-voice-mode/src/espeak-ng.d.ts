// 无类型声明的运行时依赖（espeak-ng 只发布 dist/espeak-ng.js）。
declare module 'espeak-ng' {
  interface EspeakModule {
    FS: { readFile(path: string, opts?: { encoding?: string }): string }
  }
  const factory: (opts: { arguments: string[] }) => Promise<EspeakModule>
  export default factory
}
