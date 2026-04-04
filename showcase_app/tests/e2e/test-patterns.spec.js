// @ts-check
const { test, expect } = require('@playwright/test');
const { PatternsPage } = require('./pages/PatternsPage');

test.describe('Patterns Page Interactions', () => {
  let patternsPage;

  test.beforeEach(async ({ page }) => {
    patternsPage = new PatternsPage(page);
    await patternsPage.goto();
  });

  test('should load patterns page', async ({ page }) => {
    await patternsPage.expectLoaded();
    await patternsPage.screenshot('patterns-page-loaded');
  });

  test('should display all pattern cards', async ({ page }) => {
    await expect(patternsPage.reflectionPattern).toBeVisible();
    await expect(patternsPage.planningPattern).toBeVisible();
    await expect(patternsPage.toolUsePattern).toBeVisible();
    await expect(patternsPage.agenticRAGPattern).toBeVisible();
    await expect(patternsPage.metacognitionPattern).toBeVisible();
  });

  test('should select reflection pattern', async ({ page }) => {
    await patternsPage.reflectionPattern.click();
    
    // Pattern details should be visible
    await expect(page.getByText(/self-critique|iterative improvement/i)).toBeVisible();
    
    await patternsPage.screenshot('reflection-pattern-selected');
  });

  test('should select planning pattern', async ({ page }) => {
    await patternsPage.planningPattern.click();
    
    await expect(page.getByText(/multi-step|goal decomposition/i)).toBeVisible();
  });

  test('should select tool use pattern', async ({ page }) => {
    await patternsPage.toolUsePattern.click();
    
    await expect(page.getByText(/tool selection|dynamic/i)).toBeVisible();
  });

  test('should select agentic RAG pattern', async ({ page }) => {
    await patternsPage.agenticRAGPattern.click();
    
    await expect(page.getByText(/query refinement|retrieval/i)).toBeVisible();
  });

  test('should select metacognition pattern', async ({ page }) => {
    await patternsPage.metacognitionPattern.click();
    
    await expect(page.getByText(/self-monitoring|adaptation/i)).toBeVisible();
  });

  test('should run reflection pattern', async ({ page }) => {
    await patternsPage.reflectionPattern.click();
    
    // Look for run button specific to the pattern
    const runButton = page.getByRole('button', { name: /run pattern|execute|run/i });
    await runButton.click();
    
    // Loading state should appear
    await expect(page.locator('[class*="loading"], [class*="spinner"]').first()).toBeVisible({ timeout: 3000 });
    
    // Results should appear after loading
    await expect(page.locator('[class*="output"], [class*="result"], [class*="visualization"]').first()).toBeVisible({ timeout: 30000 });
    
    await patternsPage.screenshot('reflection-pattern-results');
  });

  test('should display pattern visualization', async ({ page }) => {
    await patternsPage.reflectionPattern.click();
    
    // Run the pattern
    const runButton = page.getByRole('button', { name: /run pattern|execute|run/i });
    await runButton.click();
    
    // Wait for visualization
    await expect(page.locator('[class*="viz"], [class*="visualization"]').first()).toBeVisible({ timeout: 30000 });
    
    // Visualization should have content
    const vizContent = await page.locator('[class*="viz"], [class*="visualization"]').first().textContent();
    expect(vizContent.length).toBeGreaterThan(0);
  });

  test('should display pattern example data', async ({ page }) => {
    await patternsPage.reflectionPattern.click();
    
    // Example data should be visible
    await expect(page.getByText(/example|initial output|criteria/i)).toBeVisible();
  });

  test('should switch between patterns and see different content', async ({ page }) => {
    // Select reflection
    await patternsPage.reflectionPattern.click();
    const reflectionContent = await page.locator('[class*="pattern"]').first().textContent();
    
    // Switch to planning
    await patternsPage.planningPattern.click();
    const planningContent = await page.locator('[class*="pattern"]').first().textContent();
    
    // Content should be different
    expect(reflectionContent).not.toBe(planningContent);
  });

  test('should display pattern descriptions', async ({ page }) => {
    // Each pattern should have a description
    await expect(page.getByText(/self-critique|iterative/i)).toBeVisible();
    await expect(page.getByText(/multi-step|goal/i)).toBeVisible();
    await expect(page.getByText(/tool selection/i)).toBeVisible();
  });
});
