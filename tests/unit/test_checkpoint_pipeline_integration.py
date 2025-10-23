"""
Tests for Pipeline integration with checkpointing
"""

import pytest
from ia_modules.pipeline.core import Pipeline, Step
from ia_modules.pipeline.services import ServiceRegistry
from ia_modules.checkpoint import MemoryCheckpointer


class DummyStep(Step):
    """Simple step for testing"""

    async def run(self, data):
        step_name = self.name
        return {**data, step_name: f'completed-{step_name}'}


class TestPipelineCheckpointBasic:
    """Basic pipeline checkpoint functionality"""

    @pytest.mark.asyncio
    async def test_pipeline_without_checkpointer(self):
        """Test pipeline runs normally without checkpointer"""
        flow = {
            'start_at': 'step1',
            'paths': [
                {'from_step': 'step1', 'to_step': 'step2'},
                {'from_step': 'step2', 'to_step': 'end_with_success'}
            ]
        }
        steps = [DummyStep('step1', {}), DummyStep('step2', {})]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry())

        result = await pipeline.run({'input': 'data'})

        assert result['output']['step1'] == 'completed-step1'
        assert result['output']['step2'] == 'completed-step2'

    @pytest.mark.asyncio
    async def test_pipeline_with_checkpointer(self):
        """Test pipeline saves checkpoints when checkpointer enabled"""
        checkpointer = MemoryCheckpointer()

        flow = {
            'start_at': 'step1',
            'paths': [
                {'from_step': 'step1', 'to_step': 'step2'},
                {'from_step': 'step2', 'to_step': 'end_with_success'}
            ]
        }
        steps = [DummyStep('step1', {}), DummyStep('step2', {})]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry(), checkpointer=checkpointer)

        result = await pipeline.run({'input': 'data'}, thread_id='test-thread')

        # Check result
        assert result['output']['step1'] == 'completed-step1'
        assert result['output']['step2'] == 'completed-step2'

        # Verify checkpoints were saved
        checkpoints = await checkpointer.list_checkpoints('test-thread')
        assert len(checkpoints) == 2  # One for each step
        assert checkpoints[0].step_id == 'step2'  # Most recent
        assert checkpoints[1].step_id == 'step1'

    @pytest.mark.asyncio
    async def test_pipeline_checkpoint_state(self):
        """Test that checkpoint contains correct state"""
        checkpointer = MemoryCheckpointer()

        flow = {
            'start_at': 'step1',
            'paths': [
                {'from_step': 'step1', 'to_step': 'step2'},
                {'from_step': 'step2', 'to_step': 'end_with_success'}
            ]
        }
        steps = [DummyStep('step1', {}), DummyStep('step2', {})]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry(), checkpointer=checkpointer)

        await pipeline.run({'input': 'data'}, thread_id='test-thread')

        # Load latest checkpoint
        checkpoint = await checkpointer.load_checkpoint('test-thread')

        # Verify state
        assert 'pipeline_input' in checkpoint.state
        assert checkpoint.state['pipeline_input'] == {'input': 'data'}
        assert 'steps' in checkpoint.state
        assert 'step1' in checkpoint.state['steps']
        assert 'step2' in checkpoint.state['steps']


class TestPipelineResume:
    """Tests for pipeline resume functionality"""

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self):
        """Test resuming pipeline from checkpoint"""
        checkpointer = MemoryCheckpointer()

        flow = {
            'start_at': 'step1',
            'paths': [
                {'from_step': 'step1', 'to_step': 'step2'},
                {'from_step': 'step2', 'to_step': 'step3'},
                {'from_step': 'step3', 'to_step': 'end_with_success'}
            ]
        }
        steps = [
            DummyStep('step1', {}),
            DummyStep('step2', {}),
            DummyStep('step3', {})
        ]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry(), checkpointer=checkpointer)

        # Run pipeline but it will complete
        await pipeline.run({'input': 'data'}, thread_id='test-thread')

        # Get checkpoint after step1
        checkpoints = await checkpointer.list_checkpoints('test-thread')
        step1_checkpoint = next((cp for cp in checkpoints if cp.step_id == 'step1'), None)

        # Resume from step1 checkpoint (should execute step2 and step3)
        result = await pipeline.resume('test-thread', step1_checkpoint.checkpoint_id)

        # Verify all steps completed
        assert 'step1' in result['steps']
        assert 'step2' in result['steps']
        assert 'step3' in result['steps']

    @pytest.mark.asyncio
    async def test_resume_from_latest(self):
        """Test resuming from latest checkpoint"""
        checkpointer = MemoryCheckpointer()

        flow = {
            'start_at': 'step1',
            'paths': [
                {'from_step': 'step1', 'to_step': 'step2'},
                {'from_step': 'step2', 'to_step': 'step3'},
                {'from_step': 'step3', 'to_step': 'end_with_success'}
            ]
        }
        steps = [
            DummyStep('step1', {}),
            DummyStep('step2', {}),
            DummyStep('step3', {})
        ]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry(), checkpointer=checkpointer)

        # Run pipeline to completion
        await pipeline.run({'input': 'data'}, thread_id='test-thread')

        # Resume from latest (should be already complete)
        result = await pipeline.resume('test-thread')

        # Should return complete state
        assert result['output'] is not None

    @pytest.mark.asyncio
    async def test_resume_without_checkpointer_fails(self):
        """Test that resuming without checkpointer raises error"""
        flow = {'start_at': 'step1'}
        steps = [DummyStep('step1', {})]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry())

        with pytest.raises(ValueError, match="checkpointer not configured"):
            await pipeline.resume('test-thread')

    @pytest.mark.asyncio
    async def test_resume_nonexistent_thread_fails(self):
        """Test that resuming nonexistent thread raises error"""
        checkpointer = MemoryCheckpointer()

        flow = {'start_at': 'step1'}
        steps = [DummyStep('step1', {})]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry(), checkpointer=checkpointer)

        with pytest.raises(ValueError, match="No checkpoint found"):
            await pipeline.resume('nonexistent-thread')


class TestPipelineCheckpointThreadIsolation:
    """Tests for thread isolation in checkpointing"""

    @pytest.mark.asyncio
    async def test_multiple_threads_isolated(self):
        """Test that multiple threads have isolated checkpoints"""
        checkpointer = MemoryCheckpointer()

        flow = {
            'start_at': 'step1',
            'paths': [
                {'from_step': 'step1', 'to_step': 'end_with_success'}
            ]
        }
        steps = [DummyStep('step1', {})]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry(), checkpointer=checkpointer)

        # Run for thread1
        await pipeline.run({'thread': 1}, thread_id='thread1')

        # Run for thread2
        await pipeline.run({'thread': 2}, thread_id='thread2')

        # Load checkpoints
        cp1 = await checkpointer.load_checkpoint('thread1')
        cp2 = await checkpointer.load_checkpoint('thread2')

        # Verify isolation
        assert cp1.state['pipeline_input'] == {'thread': 1}
        assert cp2.state['pipeline_input'] == {'thread': 2}
        assert cp1.thread_id == 'thread1'
        assert cp2.thread_id == 'thread2'


class TestPipelineCheckpointMetadata:
    """Tests for checkpoint metadata"""

    @pytest.mark.asyncio
    async def test_checkpoint_metadata(self):
        """Test that checkpoint contains expected metadata"""
        checkpointer = MemoryCheckpointer()

        flow = {
            'start_at': 'step1',
            'paths': [
                {'from_step': 'step1', 'to_step': 'step2'},
                {'from_step': 'step2', 'to_step': 'end_with_success'}
            ]
        }
        steps = [DummyStep('step1', {}), DummyStep('step2', {})]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry(), checkpointer=checkpointer)

        await pipeline.run({'input': 'data'}, thread_id='test-thread')

        checkpoint = await checkpointer.load_checkpoint('test-thread')

        # Verify metadata
        assert checkpoint.pipeline_id == 'test'
        assert 'execution_path' in checkpoint.metadata
        assert isinstance(checkpoint.metadata['execution_path'], list)

    @pytest.mark.asyncio
    async def test_resumed_checkpoint_has_parent(self):
        """Test that resumed checkpoint references parent"""
        checkpointer = MemoryCheckpointer()

        flow = {
            'start_at': 'step1',
            'paths': [
                {'from_step': 'step1', 'to_step': 'step2'},
                {'from_step': 'step2', 'to_step': 'end_with_success'}
            ]
        }
        steps = [DummyStep('step1', {}), DummyStep('step2', {})]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry(), checkpointer=checkpointer)

        # Initial run
        await pipeline.run({'input': 'data'}, thread_id='test-thread')

        # Get step1 checkpoint
        checkpoints = await checkpointer.list_checkpoints('test-thread')
        step1_checkpoint = next((cp for cp in checkpoints if cp.step_id == 'step1'), None)

        # Resume from step1
        await pipeline.resume('test-thread', step1_checkpoint.checkpoint_id)

        # Get latest checkpoint
        latest = await checkpointer.load_checkpoint('test-thread')

        # Should have parent reference
        assert latest.parent_checkpoint_id == step1_checkpoint.checkpoint_id
        assert latest.metadata.get('resumed') is True


class TestPipelineCheckpointBackwardCompatibility:
    """Tests for backward compatibility"""

    @pytest.mark.asyncio
    async def test_run_without_thread_id_no_checkpoints(self):
        """Test running with checkpointer but no thread_id doesn't save checkpoints"""
        checkpointer = MemoryCheckpointer()

        flow = {'start_at': 'step1'}
        steps = [DummyStep('step1', {})]
        pipeline = Pipeline('test', steps, flow, ServiceRegistry(), checkpointer=checkpointer)

        # Run without thread_id
        result = await pipeline.run({'input': 'data'})

        # Should complete normally
        assert result['output'] is not None

        # No checkpoints should be saved
        stats = await checkpointer.get_checkpoint_stats()
        assert stats['total_checkpoints'] == 0
