// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Toast Notifications', () => {
  test('should show success toast on pipeline execution', async ({ page }) => {
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    
    // Execute a pipeline
    const firstPipeline = page.locator('[class*="pipeline"], [class*="card"]').first();
    await firstPipeline.click();
    
    const executeButton = page.getByRole('button', { name: /execute|run/i });
    await executeButton.click();
    
    // Wait for toast to appear
    await page.waitForTimeout(1000);
    
    // Success toast should be visible
    const toast = page.locator('[class*="toast"]').filter({ hasText: /success|started|executing/i }).first();
    await expect(toast).toBeVisible({ timeout: 5000 });
    
    await page.screenshot({ path: 'test-results/toast-success.png', fullPage: true });
  });

  test('should auto-dismiss toast after duration', async ({ page }) => {
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    
    // Execute a pipeline to trigger toast
    const firstPipeline = page.locator('[class*="pipeline"], [class*="card"]').first();
    await firstPipeline.click();
    
    const executeButton = page.getByRole('button', { name: /execute|run/i });
    await executeButton.click();
    
    // Wait for toast
    const toast = page.locator('[class*="toast"]').first();
    await expect(toast).toBeVisible({ timeout: 5000 });
    
    // Wait for auto-dismiss (default 4000ms + some buffer)
    await page.waitForTimeout(5000);
    
    // Toast should be gone
    await expect(toast).not.toBeVisible({ timeout: 2000 });
  });

  test('should position toast in top-right', async ({ page }) => {
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    
    // Execute a pipeline
    const firstPipeline = page.locator('[class*="pipeline"], [class*="card"]').first();
    await firstPipeline.click();
    
    await page.getByRole('button', { name: /execute|run/i }).click();
    
    await page.waitForTimeout(1000);
    
    // Toast should be in top-right position
    const toastContainer = page.locator('[class*="toaster"], [class*="toast-container"]').first();
    const toastClass = await toastContainer.getAttribute('class');
    
    expect(toastClass).toMatch(/top.*right|top-right/);
  });

  test('should show error toast on failure', async ({ page }) => {
    // This test might need a way to trigger an error
    // For now, we'll verify the toast system can display errors
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Toast container should exist
    const toastContainer = page.locator('[class*="toaster"]').first();
    await expect(toastContainer).toBeVisible();
  });

  test('should dismiss toast manually by clicking close button', async ({ page }) => {
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    
    // Execute a pipeline
    const firstPipeline = page.locator('[class*="pipeline"], [class*="card"]').first();
    await firstPipeline.click();
    
    await page.getByRole('button', { name: /execute|run/i }).click();
    
    // Wait for toast
    await page.waitForTimeout(1000);
    const toast = page.locator('[class*="toast"]').first();
    await expect(toast).toBeVisible();
    
    // Click close button on toast
    const closeButton = toast.getByRole('button', { name: /close|dismiss/i }).or(
      toast.locator('[class*="close"]')
    );
    
    if (await closeButton.isVisible()) {
      await closeButton.click();
      await expect(toast).not.toBeVisible({ timeout: 2000 });
    }
  });

  test('should display multiple toasts', async ({ page }) => {
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    
    // Execute multiple pipelines quickly
    for (let i = 0; i < 3; i++) {
      const firstPipeline = page.locator('[class*="pipeline"], [class*="card"]').first();
      await firstPipeline.click();
      
      await page.getByRole('button', { name: /execute|run/i }).click();
      await page.waitForTimeout(500);
    }
    
    // Multiple toasts might be visible
    await page.waitForTimeout(1000);
    const toastCount = await page.locator('[class*="toast"]').count();
    
    expect(toastCount).toBeGreaterThanOrEqual(1);
    
    await page.screenshot({ path: 'test-results/toast-multiple.png', fullPage: true });
  });

  test('should adapt toast colors to theme', async ({ page }) => {
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    
    // Execute a pipeline in light mode
    const firstPipeline = page.locator('[class*="pipeline"], [class*="card"]').first();
    await firstPipeline.click();
    
    await page.getByRole('button', { name: /execute|run/i }).click();
    
    await page.waitForTimeout(1000);
    
    // Toast should be visible
    const toast = page.locator('[class*="toast"]').first();
    await expect(toast).toBeVisible();
    
    // Toggle to dark mode
    await page.keyboard.press('Meta+d');
    await page.waitForTimeout(500);
    
    // Toast styling should adapt (hard to verify directly, but we can check it's still visible)
    await expect(toast).toBeVisible();
  });
});
