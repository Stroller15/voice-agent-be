import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector, elevenlabs

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.DEBUG)  # Set higher log level for better diagnostics


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarming complete, VAD model loaded")


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant. Respond **only with direct answers** to the user's question. "
            "Do **not** repeat the question, explain your history, or provide unnecessary details. "
            "Keep responses short and to the point."
        ),
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    first_response_done = False  # Track if the first response is complete

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM.with_groq(model="deepseek-r1-distill-llama-70b"),
        tts=elevenlabs.TTS(),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.2,  # Faster turn-taking
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
        allow_interruptions=False,  # Start with interruptions disabled
    )
    logger.info("Agent initialized with interruptions disabled")

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)
    logger.info("Agent started and connected to room")

    # Greet the user (interruptions disabled)
    logger.info("Sending greeting message with interruptions disabled")
    await agent.say("Hey, how can I help you today? I am Sara, your AI helper.", allow_interruptions=False)
    
    # Double-check interruptions state after greeting
    logger.info(f"After greeting: interruptions enabled = {agent.allow_interruptions}")

    # Handle user input
    logger.info("Starting user interaction loop")
    async for text in agent.listen():
        logger.info(f"Received user input: '{text}'")
        logger.info(f"Current interruption state before processing: {agent.allow_interruptions}")

        if not first_response_done:
            # First response: explicitly disable interruptions
            agent.allow_interruptions = False
            logger.info("Processing first response with interruptions DISABLED")

            try:
                response = await agent.respond(text)
                logger.info(f"First response generated: '{response[:50]}...'")
                
                # Explicitly disable interruptions for the first response
                await agent.say(response, allow_interruptions=False)
                logger.info("First response delivered successfully")

                # Mark first response as complete
                first_response_done = True

                # NOW enable interruptions for future interactions
                agent.allow_interruptions = True
                logger.info("First response complete. Interruptions now ENABLED for future interactions")
                logger.info(f"Updated interruption state: {agent.allow_interruptions}")

            except Exception as e:
                logger.error(f"Error during first response: {e}")
                first_response_done = True  # Mark as done anyway to prevent getting stuck
                agent.allow_interruptions = True  # Ensure interruptions are enabled
        else:
            # Subsequent interactions - interruptions should be enabled
            logger.info("Processing subsequent input with interruptions ENABLED")

            # Ensure interruptions are enabled
            if not agent.allow_interruptions:
                logger.warning("Interruptions were disabled but should be enabled - fixing")
                agent.allow_interruptions = True

            # Stop any ongoing speech (without blocking)
            try:
                logger.info("Stopping any ongoing speech")
                await agent.tts.stop()
                logger.info("Successfully stopped ongoing speech")
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")

            try:
                # Generate and deliver response with interruptions enabled
                response = await agent.respond(text)
                logger.info(f"Subsequent response generated: '{response[:50]}...'")

                await agent.say(response, allow_interruptions=True)
                logger.info("Response delivered successfully with interruptions enabled")
                logger.info(f"Current interruption state after delivery: {agent.allow_interruptions}")

            except Exception as e:
                logger.error(f"Error during subsequent response: {e}")


if __name__ == "__main__":
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )