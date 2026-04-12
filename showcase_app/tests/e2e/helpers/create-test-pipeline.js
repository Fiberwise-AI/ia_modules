// @ts-check

/**
 * Creates a test pipeline via the API (more reliable than UI)
 * @param {import('@playwright/test').APIRequestContext} request
 * @returns {Promise<{success: boolean, pipelineId?: string, name?: string}>}
 */
async function createTestPipeline(request) {
  const API_BASE = 'http://localhost:7331';

  const pipelineData = {
    name: "E2E Test Pipeline",
    description: "Created by e2e test suite",
    version: "1.0",
    steps: [
      {
        id: "step1",
        name: "step1",
        step_class: "TransformStep",
        module: "pipelines.examples",
        config: { operation: "uppercase" }
      },
      {
        id: "step2",
        name: "step2",
        step_class: "TransformStep",
        module: "pipelines.examples",
        config: { operation: "reverse" }
      }
    ],
    connections: [
      { from: "step1", to: "step2" }
    ],
    metadata: {
      created_by: "e2e-tests"
    }
  };

  try {
    const response = await request.post(`${API_BASE}/api/pipelines`, {
      data: pipelineData
    });

    if (response.ok()) {
      const body = await response.json();
      return {
        success: true,
        pipelineId: body.id,
        name: body.name
      };
    }

    // If POST fails, try to find existing pipelines
    const listResponse = await request.get(`${API_BASE}/api/pipelines`);
    if (listResponse.ok()) {
      const pipelines = await listResponse.json();
      if (Array.isArray(pipelines) && pipelines.length > 0) {
        return {
          success: true,
          pipelineId: pipelines[0].id,
          name: pipelines[0].name
        };
      }
    }

    return { success: false };
  } catch (error) {
    console.error('Failed to create test pipeline:', error.message);
    return { success: false };
  }
}

module.exports = { createTestPipeline };
