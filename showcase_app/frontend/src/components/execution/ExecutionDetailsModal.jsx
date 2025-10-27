import { useState } from 'react';
import { X, CheckCircle, XCircle, Clock, ChevronDown, ChevronRight, AlertCircle } from 'lucide-react';
import { hitlAPI } from '../../services/api';
import toast from 'react-hot-toast';

export default function ExecutionDetailsModal({ execution, hitlInteraction, onClose }) {
  const [expandedSteps, setExpandedSteps] = useState({});
  const [formData, setFormData] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);


  const toggleStep = (stepName) => {
    setExpandedSteps(prev => ({
      ...prev,
      [stepName]: !prev[stepName]
    }));
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-500" size={20} />;
      case 'failed':
        return <XCircle className="text-red-500" size={20} />;
      default:
        return <Clock className="text-blue-500" size={20} />;
    }
  };

  const handleFieldChange = (fieldName, value) => {
    setFormData(prev => ({
      ...prev,
      [fieldName]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      await hitlAPI.respond(hitlInteraction.interaction_id, formData);
      alert('Response submitted successfully! Pipeline will resume.');
      onClose();
      // Refresh page to see updated status
      window.location.reload();
    } catch (error) {
      alert(`Error submitting response: ${error.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderField = (field) => {
    const value = formData[field.name] || field.default || '';

    switch (field.type) {
      case 'text':
        return (
          <input
            type="text"
            value={value}
            onChange={(e) => handleFieldChange(field.name, e.target.value)}
            placeholder={field.placeholder}
            required={field.required}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-yellow-500"
          />
        );

      case 'textarea':
        return (
          <textarea
            value={value}
            onChange={(e) => handleFieldChange(field.name, e.target.value)}
            placeholder={field.placeholder}
            required={field.required}
            rows={field.rows || 3}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-yellow-500"
          />
        );

      case 'radio':
        return (
          <div className="space-y-2">
            {field.options.map((option) => {
              const optionValue = typeof option === 'object' ? option.value : option;
              const optionLabel = typeof option === 'object' ? option.label : option;

              return (
                <label key={optionValue} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name={field.name}
                    value={optionValue}
                    checked={value === optionValue}
                    onChange={(e) => handleFieldChange(field.name, e.target.value)}
                    required={field.required}
                    className="w-4 h-4 text-yellow-600"
                  />
                  <span className="text-sm">{optionLabel}</span>
                </label>
              );
            })}
          </div>
        );

      case 'checkbox':
        return (
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={value === true}
              onChange={(e) => handleFieldChange(field.name, e.target.checked)}
              required={field.required}
              className="w-4 h-4 text-yellow-600 rounded"
            />
            <span className="text-sm">{field.label}</span>
          </label>
        );

      default:
        return <p className="text-sm text-gray-500">Unsupported field type: {field.type}</p>;
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Execution Details</h2>
            <p className="text-sm text-gray-500 font-mono mt-1">{execution.job_id}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X size={24} />
          </button>
        </div>


        {/* Status Summary */}
        <div className="p-6 border-b bg-gray-50">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-gray-500">Status</p>
              <div className="flex items-center gap-2 mt-1">
                {getStatusIcon(execution.status)}
                <span className="font-semibold capitalize">{execution.status.replace('_', ' ')}</span>
              </div>
            </div>
            <div>
              <p className="text-xs text-gray-500">Duration</p>
              <p className="font-semibold mt-1">
                {execution.execution_time_ms ? `${(execution.execution_time_ms / 1000).toFixed(2)}s` : '-'}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Steps</p>
              <p className="font-semibold mt-1">
                {execution.completed_steps || 0} / {execution.total_steps || 0}
              </p>
            </div>
          </div>
          {execution.error_message && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded">
              <p className="text-sm font-semibold text-red-900">Error</p>
              <p className="text-sm text-red-700 mt-1">{execution.error_message}</p>
            </div>
          )}
        </div>

        {/* Steps */}
        <div className="flex-1 overflow-y-auto p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Pipeline Results</h3>
          {(!execution.steps || execution.steps.length === 0) && execution.output_data && (
            <div className="mb-6">
              <p className="text-sm text-gray-500 mb-2">Final Output:</p>
              <pre className="bg-gray-50 border rounded p-4 text-xs overflow-x-auto max-h-96">
                {JSON.stringify(execution.output_data, null, 2)}
              </pre>
            </div>
          )}
          {(!execution.steps || execution.steps.length === 0) && !execution.output_data && (
            <p className="text-sm text-gray-500">No execution data available.</p>
          )}
          <div className="space-y-2">
            {execution.steps?.map((step, index) => {
              const isHITLStep = hitlInteraction && hitlInteraction.step_name === step.step_name;

              return (
              <div key={index} className={`border rounded-lg overflow-hidden ${isHITLStep ? 'border-yellow-400 bg-yellow-50' : ''}`}>
                <button
                  onClick={() => toggleStep(step.step_name)}
                  className={`w-full flex items-center justify-between p-4 hover:bg-gray-50 transition ${isHITLStep ? 'bg-yellow-50' : 'bg-white'}`}
                >
                  <div className="flex items-center gap-3">
                    {expandedSteps[step.step_name] ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    {isHITLStep ? <AlertCircle className="text-yellow-600" size={20} /> : getStatusIcon(step.status)}
                    <div className="text-left">
                      <p className="font-semibold text-gray-900 flex items-center gap-2">
                        {step.step_name}
                        {isHITLStep && <span className="text-xs bg-yellow-500 text-white px-2 py-0.5 rounded">Approval Required</span>}
                      </p>
                      {step.duration_ms && (
                        <p className="text-xs text-gray-500">{(step.duration_ms / 1000).toFixed(2)}s</p>
                      )}
                    </div>
                  </div>
                </button>

                {expandedSteps[step.step_name] && (
                  <div className="p-4 bg-gray-50 border-t space-y-4">
                    {/* HITL Approval Form */}
                    {isHITLStep && (
                      <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4 mb-4">
                        <p className="text-sm font-semibold text-yellow-900 mb-3">{hitlInteraction.prompt}</p>

                        {/* Show non-decision fields (like comments) */}
                        {hitlInteraction.ui_schema?.fields?.filter(f => f.name !== 'decision' && f.type !== 'radio').map((field, idx) => (
                          <div key={idx} className="mb-3">
                            <label className="block text-xs font-medium text-gray-700 mb-1">
                              {field.label} {field.required && <span className="text-red-500">*</span>}
                            </label>
                            {renderField(field)}
                          </div>
                        ))}

                        {/* Approve/Reject Buttons */}
                        <div className="flex gap-2 mt-3">
                          <button
                            type="button"
                            disabled={isSubmitting}
                            onClick={async (e) => {
                              e.preventDefault();
                              setFormData(prev => ({ ...prev, decision: 'approve' }));
                              setIsSubmitting(true);
                              try {
                                await hitlAPI.respond(hitlInteraction.interaction_id, { ...formData, decision: 'approve' }, null);
                                toast.success('Approved! Pipeline is resuming...');
                                onClose();
                              } catch (error) {
                                toast.error(`Error: ${error.message}`);
                                setIsSubmitting(false);
                              }
                            }}
                            className="flex-1 px-3 py-2 bg-green-500 text-white text-sm rounded hover:bg-green-600 font-semibold disabled:opacity-50"
                          >
                            {isSubmitting ? 'Submitting...' : '✓ Approve'}
                          </button>
                          <button
                            type="button"
                            disabled={isSubmitting}
                            onClick={async (e) => {
                              e.preventDefault();
                              setFormData(prev => ({ ...prev, decision: 'reject' }));
                              setIsSubmitting(true);
                              try {
                                await hitlAPI.respond(hitlInteraction.interaction_id, { ...formData, decision: 'reject' }, null);
                                toast.success('Rejected! Pipeline is resuming...');
                                onClose();
                              } catch (error) {
                                toast.error(`Error: ${error.message}`);
                                setIsSubmitting(false);
                              }
                            }}
                            className="flex-1 px-3 py-2 bg-red-500 text-white text-sm rounded hover:bg-red-600 font-semibold disabled:opacity-50"
                          >
                            {isSubmitting ? 'Submitting...' : '✗ Reject'}
                          </button>
                        </div>
                      </div>
                    )}
                    {/* Input Data */}
                    {step.input_data && Object.keys(step.input_data).length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-gray-700 mb-2">Input Data</p>
                        <pre className="bg-white border rounded p-3 text-xs overflow-x-auto">
                          {JSON.stringify(step.input_data, null, 2)}
                        </pre>
                      </div>
                    )}

                    {/* Output Data */}
                    {step.output_data && Object.keys(step.output_data).length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-gray-700 mb-2">Output Data</p>
                        <pre className="bg-white border rounded p-3 text-xs overflow-x-auto">
                          {JSON.stringify(step.output_data, null, 2)}
                        </pre>
                      </div>
                    )}

                    {/* Error */}
                    {step.error && (
                      <div>
                        <p className="text-xs font-semibold text-red-700 mb-2">Error</p>
                        <pre className="bg-red-50 border border-red-200 rounded p-3 text-xs text-red-700 overflow-x-auto">
                          {step.error}
                        </pre>
                      </div>
                    )}

                    {/* Logs */}
                    {step.logs && step.logs.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-gray-700 mb-2">Logs</p>
                        <div className="bg-white border rounded p-3 space-y-1 max-h-48 overflow-y-auto">
                          {step.logs.map((log, i) => (
                            <p key={i} className="text-xs font-mono text-gray-700">{log}</p>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
              );
            })}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t bg-gray-50">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
