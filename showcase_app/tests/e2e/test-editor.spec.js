// @ts-check
const { test, expect } = require('@playwright/test');
const { PipelineEditorPage } = require('./pages/PipelineEditorPage');

test.describe('Pipeline Editor', () => {
  let editorPage;

  test.beforeEach(async ({ page }) => {
    editorPage = new PipelineEditorPage(page);
    await editorPage.goto();
  });

  test('should load pipeline editor', async ({ page }) => {
    await editorPage.expectLoaded();
    await editorPage.screenshot('editor-page-loaded');
  });

  test('should display view mode tabs', async ({ page }) => {
    await expect(editorPage.visualTab).toBeVisible();
    await expect(editorPage.codeTab).toBeVisible();
    await expect(editorPage.splitTab).toBeVisible();
  });

  test('should switch to code view mode', async ({ page }) => {
    await editorPage.switchToCodeMode();
    
    // Code editor should be visible
    await expect(editorPage.codeEditor).toBeVisible();
    
    await editorPage.screenshot('editor-code-view');
  });

  test('should switch to split view mode', async ({ page }) => {
    await editorPage.switchToSplitMode();
    
    // Both canvas and code editor should be visible
    await expect(editorPage.codeEditor).toBeVisible();
    
    await editorPage.screenshot('editor-split-view');
  });

  test('should display toolbar buttons', async ({ page }) => {
    await expect(editorPage.saveButton).toBeVisible();
    await expect(editorPage.runButton).toBeVisible();
    await expect(editorPage.loadButton).toBeVisible();
    await expect(editorPage.backButton).toBeVisible();
  });

  test('should open load pipeline dialog', async ({ page }) => {
    await editorPage.openLoadDialog();
    
    // Dialog should show pipeline list
    await expect(editorPage.loadDialogPipelineList).toHaveCount({ min: 1 });
    
    await editorPage.screenshot('editor-load-dialog');
  });

  test('should load existing pipeline', async ({ page }) => {
    // Open load dialog and select first pipeline
    await editorPage.openLoadDialog();
    
    const firstPipeline = editorPage.loadDialogPipelineList.first();
    await firstPipeline.click();
    
    // Dialog should close and pipeline should load
    await expect(editorPage.loadDialog).not.toBeVisible();
    
    await editorPage.screenshot('editor-pipeline-loaded');
  });

  test('should display executions table after loading pipeline', async ({ page }) => {
    // Load a pipeline first
    await editorPage.openLoadDialog();
    const firstPipeline = editorPage.loadDialogPipelineList.first();
    await firstPipeline.click();
    
    // Wait for executions table to appear
    await expect(editorPage.executionsTable).toBeVisible({ timeout: 10000 });
  });

  test('should run pipeline from editor', async ({ page }) => {
    // Load a pipeline
    await editorPage.openLoadDialog();
    const firstPipeline = editorPage.loadDialogPipelineList.first();
    await firstPipeline.click();
    
    // Wait a moment for pipeline to load
    await page.waitForTimeout(1000);
    
    // Click run button
    await editorPage.runPipeline.click();
    
    // Execution modal or dialog should appear
    await expect(page.getByRole('dialog', { name: /run|execute|input/i })).toBeVisible({ timeout: 5000 });
  });

  test('should show visual canvas in visual mode', async ({ page }) => {
    // Ensure we're in visual mode
    await editorPage.switchToVisualMode();
    
    // Visual canvas should be present (check for ReactFlow or similar)
    const visualCanvas = page.locator('[class*="reactflow"], [class*="canvas"], svg').first();
    await expect(visualCanvas).toBeVisible({ timeout: 5000 });
  });

  test('should track unsaved changes', async ({ page }) => {
    // Load a pipeline
    await editorPage.openLoadDialog();
    const firstPipeline = editorPage.loadDialogPipelineList.first();
    await firstPipeline.click();
    
    // Wait for pipeline to load
    await page.waitForTimeout(1000);
    
    // Switch to code mode and make changes
    await editorPage.switchToCodeMode();
    
    // Edit the code editor (simulate typing)
    await editorPage.codeEditor.click();
    await page.keyboard.press('End');
    await page.keyboard.type('  // modified');
    
    // Save button might show unsaved indicator
    await page.waitForTimeout(500);
    
    await editorPage.screenshot('editor-unsaved-changes');
  });

  test('should navigate back to pipelines', async ({ page }) => {
    await editorPage.backButton.click();
    
    // Should navigate to pipelines page
    await page.waitForURL('**/pipelines');
    await expect(page.getByRole('heading', { name: /pipelines/i })).toBeVisible();
  });
});
