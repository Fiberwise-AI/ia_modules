// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Execution Detail Page', () => {
  test('should load execution detail page with valid job ID', async ({ page }) => {
    // First, execute a pipeline to get a job ID
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    
    // Execute a simple pipeline
    const firstPipeline = page.locator('[class*="pipeline"], [class*="card"]').first();
    await firstPipeline.click();
    
    const executeButton = page.getByRole('button', { name: /execute|run/i });
    await executeButton.click();
    
    // Wait for execution to start
    await page.waitForTimeout(2000);
    
    // Navigate to executions page to get job ID
    await page.goto('http://localhost:5173/executions');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    // Get first job ID from the table
    const firstJobIdCell = page.locator('tbody td').nth(1);
    const jobId = await firstJobIdCell.textContent();
    
    if (jobId) {
      // Navigate to execution detail
      await page.goto(`http://localhost:5173/executions/${jobId}`);
      await page.waitForLoadState('networkidle');
      
      // Page should load successfully
      await expect(page.getByRole('heading', { name: /execution/i })).toBeVisible({ timeout: 5000 });
      
      await page.screenshot({ path: 'test-results/execution-detail-loaded.png', fullPage: true });
    }
  });

  test('should display step details', async ({ page }) => {
    // Navigate to executions
    await page.goto('http://localhost:5173/executions');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    // Get first job ID
    const firstJobIdCell = page.locator('tbody td').nth(1);
    const jobId = await firstJobIdCell.textContent();
    
    if (jobId) {
      await page.goto(`http://localhost:5173/executions/${jobId}`);
      await page.waitForLoadState('networkidle');
      
      // Step details should be visible
      await expect(page.locator('[class*="step"], [class*="detail"]').first()).toBeVisible({ timeout: 5000 });
    }
  });

  test('should display telemetry timeline', async ({ page }) => {
    await page.goto('http://localhost:5173/executions');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    const firstJobIdCell = page.locator('tbody td').nth(1);
    const jobId = await firstJobIdCell.textContent();
    
    if (jobId) {
      await page.goto(`http://localhost:5173/executions/${jobId}`);
      await page.waitForLoadState('networkidle');
      
      // Telemetry timeline should be present
      const hasTimeline = await page.locator('[class*="telemetry"], [class*="timeline"]').first().isVisible().catch(() => false);
      expect(hasTimeline).toBe(true);
    }
  });

  test('should display input/output data viewers', async ({ page }) => {
    await page.goto('http://localhost:5173/executions');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    const firstJobIdCell = page.locator('tbody td').nth(1);
    const jobId = await firstJobIdCell.textContent();
    
    if (jobId) {
      await page.goto(`http://localhost:5173/executions/${jobId}`);
      await page.waitForLoadState('networkidle');
      
      // Data viewers should be present
      const hasDataViewer = await page.locator('[class*="data-viewer"], [class*="input"], [class*="output"]').first().isVisible().catch(() => false);
      expect(hasDataViewer).toBe(true);
    }
  });

  test('should display pipeline graph', async ({ page }) => {
    await page.goto('http://localhost:5173/executions');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    const firstJobIdCell = page.locator('tbody td').nth(1);
    const jobId = await firstJobIdCell.textContent();
    
    if (jobId) {
      await page.goto(`http://localhost:5173/executions/${jobId}`);
      await page.waitForLoadState('networkidle');
      
      // Pipeline graph should be visible
      const hasGraph = await page.locator('[class*="graph"], [class*="flow"]').first().isVisible().catch(() => false);
      expect(hasGraph).toBe(true);
    }
  });

  test('should handle invalid job ID gracefully', async ({ page }) => {
    await page.goto('http://localhost:5173/executions/nonexistent-job-id');
    await page.waitForLoadState('networkidle');
    
    // Should show error state or not found message
    const hasError = await page.getByText(/not found|error|invalid/i).isVisible().catch(() => false);
    expect(hasError).toBe(true);
  });
});
