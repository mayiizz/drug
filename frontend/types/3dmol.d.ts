declare module "3dmol/build/3Dmol.js" {
  export interface ViewerStyle {
    cartoon?: { color?: string; colorfunc?: (atom: { b?: number; ss?: string }) => string; opacity?: number };
    stick?: { radius?: number; color?: string; colorscheme?: string };
    sphere?: { scale?: number; color?: string; opacity?: number };
    surface?: { opacity?: number; color?: string };
  }

  export interface GLViewer {
    addModel(data: string, format: string): unknown;
    setStyle(sel: object, style: ViewerStyle): void;
    addStyle(sel: object, style: ViewerStyle): void;
    setBackgroundColor(color: string | number): void;
    zoomTo(sel?: object): void;
    zoom(factor: number, animationDuration?: number): void;
    render(): void;
    clear(): void;
    removeAllModels(): void;
    removeAllSurfaces?: () => void;
    addSurface(
      type: unknown,
      style: { opacity?: number; color?: string; colorscheme?: string },
      sel?: object
    ): Promise<unknown> | unknown;
    resize?: () => void;
    getCanvas(): HTMLCanvasElement;
  }

  export const SurfaceType: { VDW: unknown; MS: unknown; SAS: unknown; SES: unknown };

  export function createViewer(
    element: HTMLElement,
    config?: { backgroundColor?: string; id?: string }
  ): GLViewer;

  const $3Dmol: {
    createViewer: typeof createViewer;
    SurfaceType: typeof SurfaceType;
  };

  export default $3Dmol;
}
