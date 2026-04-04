// @ts-check
const { test, expect } = require('@playwright/test');
const { PipelinesPage } = require('./pages/PipelinesPage');

test.describe('Pipeline Execution', () => {
  let pipelinesPage;

  test.beforeEach(async ({ page }) => {
    pipelinesPage = new PipelinesPage(page);
    await pipelinesPage.goto();
  });

  test('should load pipelines page', async ({ page }) => {
    await pipelinesPage.expectLoaded();
    await pipelinesPage.screenshot('pipelines-page-loaded');
  });

  test('should display pipeline list', async ({ page }) => {
    await expect(pipelinesPage.pipelineList).toBeVisible();
  });

  test('should execute a simple pipeline', async ({ page }) => {
    // Click on simple pipeline
    await pipelinesPage.simplePipeline.click();
    
    // Execute pipeline
    await pipelinesPage.executeButton.click();
    
    // Wait for execution to complete
    await pipelinesPage.waitForExecutionComplete();
    
    // Verify output is displayed
    await expect(pipelinesPage.outputSection).toBeVisible({ timeout: 30000 });
    
    await pipelinesPage.screenshot('pipeline-execution-complete');
  });

  test('should display pipeline steps', async ({ page }) => {
    // Execute any pipeline to see steps
    await pipelinesPage.simplePipeline.click();
    await pipelinesPage.executeButton.click();
    
    // Wait for steps to appear
    await expect(pipelinesPage.stepList).toHaveCount({ min: 1 }, { timeout: 30000 });
  });

  test('should show execution status', async ({ page }) => {
    await pipelinesPage.simplePipeline.click();
    await pipelinesPage.executeButton.click();
    
    // Wait for execution status to appear
    await expect(pipelinesPage.executionStatus).toBeVisible({ timeout: 30000 });
  });

  test('should handle conditional pipeline', async ({ page }) => {
    // Click on conditional pipeline
    await pipelinesPage.conditionalPipeline.click();
    await pipelinesPage.executeButton.click();
    
    // Wait for execution
    await pipelinesPage.waitForExecutionComplete();
    
    await pipelinesPage.screenshot('conditional-pipeline-complete');
  });

  test('should handle parallel pipeline', async ({ page }) => {
    await pipelinesPage.parallelPipeline.click();
    await pipelinesPage.executeButton.click();
    
    await pipelinesPage.waitForExecutionComplete();
    await expect(pipelinesPage.outputSection).toBeVisible({ timeout: 30000 });
  });

  test('should display pipeline details after execution', async ({ page }) => {
    await pipelinesPage.simplePipeline.click();
    await pipelinesPage.executeButton.click();
    
    // Wait for completion
    await pipelinesPage.waitForExecutionComplete();
    
    // Verify execution details are shown
    await expect(page.getByText(/complete|success|finished/i)).toBeVisible({ timeout: 30000 });
  });

  test('should handle multiple pipeline executions', async ({ page }) => {
    // Execute pipeline multiple times
    for (let i = 0; i < 3; i++) {
      await pipelinesPage.simplePipeline.click();
      await pipelinesPage.executeButton.click();
      await pipelinesPage.waitForExecutionComplete();
    }
    
    await pipelinesPage.screenshot('multiple-executions');
  });
});
