import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from agent import WeatherAgent

logger = logging.getLogger(__name__)

class WeatherAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = WeatherAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 1. 验证请求 (此处简化)
        
        # 2. 获取或创建 Task
        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
            
        # 3. 初始化状态更新器
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            # 4. 消费 Agent 的流
            async for item in self.agent.stream(query, task.context_id):
                is_complete = item['is_task_complete']
                require_input = item['require_user_input']
                content = item['content']

                # 情况 A: 正在工作中 (Working)
                if not is_complete and not require_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(content, task.context_id, task.id)
                    )
                
                # 情况 B: 需要用户补充信息 (Input Required)
                elif require_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(content, task.context_id, task.id),
                        final=True # 暂停 Task 等待输入
                    )
                    break
                
                # 情况 C: 任务完成 (Completed)
                else:
                    # 添加最终结果 Artifact
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name='weather_result',
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f"Error executing agent: {e}")
            raise ServerError(error=InvalidParamsError(message=str(e)))

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())