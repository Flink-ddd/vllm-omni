import asyncio
import pytest
import torch
import uuid
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.diffusion.data import OmniACK
from vllm_omni.inputs.data import OmniSamplingParams

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniTest")

pytestmark = [pytest.mark.core_model, pytest.mark.gpu]

@pytest.fixture(scope="module")
async def omni_engine():
    model_name = "Qwen/Qwen2.5-Omni-3B"
    stage_0 = {
        "stage_id": 0,
        "stage_type": "llm",
        "runtime": {"process": True, "devices": "0", "max_batch_size": 1},
        "engine_args": {
            "model": model_name,
            "model_stage": "thinker",
            "enable_sleep_mode": True,
            "quantization": "fp8",
            "enforce_eager": True,
        }
    }
    engine = AsyncOmni(model_name, stages=[stage_0])
    for stage in engine.stage_list:
        if hasattr(stage, "engine") and stage.engine:
            stage.engine.orchestrator = engine
    yield engine
    engine.shutdown()


class TestOmniSleepMode:

    @pytest.mark.asyncio
    async def test_model_integrity_post_wakeup(self, omni_engine: AsyncOmni):
        """
        Verify the consistency of model output after Sleep/Wakeup (to prevent garbled characters)
        Verification points: Text consistency & Exact matching of Token IDs
        """
        logger.info("Starting Consistency Test: Baseline vs Post-Wakeup")
        prompt = "The capital of France is"
        sampling_params = OmniSamplingParams(max_tokens=10, temperature=0.0)

        # Baseline Generation
        logger.info("Running Baseline Generation...")
        base_text = ""
        base_ids = []
        async for output in omni_engine.generate(prompt=prompt, request_id="baseline", sampling_params_list=[sampling_params]):
            res = output.request_output.outputs[0]
            base_text = res.text
            base_ids = res.token_ids

        logger.info(f"Baseline IDs: {base_ids}")
        logger.info(f"Baseline Text: '{base_text}'")

        # Trigger Deep Sleep & Wakeup Cycle
        logger.info("Testing Sleep Level 2 & Deterministic Wakeup...")
        await omni_engine.sleep(stage_ids=[0], level=2)
        logger.info("Engine is SLEEPING. VRAM should be released.")

        await asyncio.sleep(2)
        await omni_engine.wake_up(stage_ids=[0])
        logger.info("Engine is WAKEN UP. Weights restored.")

        logger.info("Running Post-Wakeup Generation...")
        
        post_text = ""
        post_ids = []
        async for output in omni_engine.generate(prompt=prompt, request_id="post-wake", sampling_params_list=[sampling_params]):
            res = output.request_output.outputs[0]
            post_text = res.text
            post_ids = res.token_ids

        logger.info("Comparing Results...")
        try:
            assert base_ids == post_ids, f"Token IDs Mismatch!\nBase: {base_ids}\nPost: {post_ids}"
            assert base_text == post_text, f"Text Mismatch!\nBase: {base_text}\nPost: {post_text}"
            logger.info("SUCCESS: Model output is identical (Bit-Identical)!")
            logger.info("FP8 scaling factors and weights are verified correct.")
        except AssertionError as e:
            logger.error(f"FAIL: Consistency Check Failed!")
            logger.error(f"Last 3 IDs (Base): {base_ids[-3:]}")
            logger.error(f"Last 3 IDs (Post): {post_ids[-3:]}")
            raise e

    @pytest.mark.asyncio
    async def test_task1_physical_reclamation(self, omni_engine: AsyncOmni):
        """Verify physical memory reclamation"""
        logger.info("Starting Task 1 Test: Physical Reclamation")
        initial_mem = torch.cuda.memory_reserved()
        logger.info(f"Initial VRAM Reserved: {initial_mem / 1024**3:.2f} GiB")
        # Trigger deep sleep (Level 2)
        # Verification: await must return only after all Workers have finished moving GPU memory.
        acks = await omni_engine.sleep(stage_ids=[0], level=2)
        post_sleep_mem = torch.cuda.memory_reserved()
        freed_gb = (initial_mem - post_sleep_mem) / 1024**3
        logger.info(f"Post-Sleep VRAM Reserved: {post_sleep_mem / 1024**3:.2f} GiB")
        logger.info(f"Total Freed: {freed_gb:.2f} GiB")
        assert freed_gb > 20.0, f"VRAM reclamation failed, only freed {freed_gb:.2f} GiB"
        assert all(ack.status == "SUCCESS" for ack in acks)

    @pytest.mark.asyncio
    async def test_task2_deterministic_handshake(self, omni_engine: AsyncOmni):
        """Verification of multi-worker signal aggregation and interception"""
        logger.info("Starting Task 2 Test: Deterministic Handshake")
        task_id = str(uuid.uuid4())
        stage = omni_engine.stage_list[0]
        # Verification: The Resolver can correctly count the number of Workers.
        expected_count = stage.engine.executor.get_worker_count()
        future = omni_engine.resolver.watch_task(task_id, expected_count=expected_count)
        stage.sleep(level=2, task_id=task_id)
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.wait_for(future, timeout=30.0)
        duration = asyncio.get_event_loop().time() - start_time
        logger.info(f"Handshake resolved in {duration:.2f}s with {len(results)} ACKs")
        assert len(results) == expected_count, "ACK aggregation mismatch!"

    @pytest.mark.asyncio
    async def test_task3_auto_wakeup_protection(self, omni_engine: AsyncOmni):
        """Verify automatic wake-up protection logic"""
        logger.info("Starting Task 3 Test: Auto-Wakeup Protection")
        # First, let the stage enter sleep.
        await omni_engine.sleep(stage_ids=[0], level=2)
        assert omni_engine.stage_list[0].status == "SLEEPING"
        # Attempt to send an inference request while in the SLEEPING state.
        prompt = "A high-tech lab in Kuala Lumpur at night"
        async for output in omni_engine.generate(prompt=prompt, request_id="test-auto-wake"):
            assert output is not None
            break  # Only the first output needs to be taken.
        # The final confirmation phase has been activated.
        assert omni_engine.stage_list[0].status == "RUNNING"
        logger.info("Auto-Wakeup verified successfully.")


    @pytest.mark.asyncio
    async def test_error_fallback_and_timeout(self, omni_engine: AsyncOmni):
        """Exception and Timeout Handling"""
        logger.info("Starting Scenario 4: Timeout Handling")
        try:
            await asyncio.wait_for(omni_engine.sleep(level=2), timeout=0.001)
        except asyncio.TimeoutError:
            logger.info("Timeout handled correctly.")