# coding=utf-8
"""
    @project: MaxKB
    @Author：虎虎
    @file： chat.py
    @date：2025/6/6 11:18
    @desc:
"""
import requests
import time
import json

from django.db.models import QuerySet
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.utils.translation import gettext_lazy as _
from drf_spectacular.utils import extend_schema
from rest_framework.parsers import MultiPartParser
from rest_framework.request import Request
from rest_framework.views import APIView

from application.api.application_api import SpeechToTextAPI, TextToSpeechAPI
from application.models import Application
from chat.api.chat_api import ChatAPI
from chat.api.chat_authentication_api import ChatAuthenticationAPI, ChatAuthenticationProfileAPI, ChatOpenAPI, OpenAIAPI
from chat.serializers.chat import OpenChatSerializers, ChatSerializers, SpeechToTextSerializers, \
    TextToSpeechSerializers, OpenAIChatSerializer, CompletionsRequestSerializer, CompletionsResponseSerializer
from chat.serializers.chat_authentication import AnonymousAuthenticationSerializer, ApplicationProfileSerializer, \
    AuthProfileSerializer
from common.auth import ChatTokenAuth
from common.config.tokenizer_manage_config import TokenizerManage
from common.constants.permission_constants import ChatAuth
from common.exception.app_exception import AppAuthenticationFailed
from common.result import result
from knowledge.models import FileSourceType
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from models_provider.models import Model
from models_provider.tools import get_model_instance_by_model_workspace_id
from oss.serializers.file import FileSerializer
from users.api import CaptchaAPI
from users.serializers.login import CaptchaSerializer
from common.utils.logger import maxkb_logger


def _mask_headers(headers: dict) -> dict:
    safe_headers = dict(headers)
    for k in list(safe_headers.keys()):
        if str(k).lower() in {'authorization', 'proxy-authorization'}:
            safe_headers[k] = 'Bearer ****'
    return safe_headers


def _safe_jsonable(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8', errors='replace')
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if hasattr(value, 'lists'):
        try:
            return {str(k): [_safe_jsonable(vv) for vv in v] for k, v in value.lists()}
        except Exception:
            return str(value)
    if hasattr(value, 'name') and hasattr(value, 'size'):
        try:
            return {'file_name': getattr(value, 'name', ''), 'file_size': getattr(value, 'size', None)}
        except Exception:
            return str(value)
    return str(value)


def _log_incoming_request(request: Request):
    try:
        headers = _mask_headers(dict(request.headers.items()))
        remote_addr = request.META.get('HTTP_X_FORWARDED_FOR') or request.META.get('REMOTE_ADDR')
        payload = {
            'path': request.path,
            'method': request.method,
            'remote_addr': remote_addr,
            'headers': headers,
            'query': _safe_jsonable(request.query_params),
            'body': _safe_jsonable(request.data),
        }
        msg = json.dumps(payload, ensure_ascii=False)
        maxkb_logger.warning(msg)
        print(msg, flush=True)
        logging.info(f"FAY_REQUEST: {msg}")
    except Exception:
        pass


def stream_image(response):
    """生成器函数，用于流式传输图片数据"""
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:  # 过滤掉保持连接的空块
            yield chunk


class ResourceProxy(APIView):
    def get(self, request: Request):
        image_url = request.query_params.get("url")
        if not image_url:
            return result.error("Missing 'url' parameter")
        try:

            # 发送GET请求，流式获取图片内容
            response = requests.get(
                image_url,
                stream=True,  # 启用流式响应
                allow_redirects=True,
                timeout=10
            )
            content_type = response.headers.get('Content-Type', '').split(';')[0]
            # 创建Django流式响应
            django_response = StreamingHttpResponse(
                stream_image(response),  # 使用生成器
                content_type=content_type
            )

            return django_response
        except Exception as e:
            return result.error(f"Image request failed: {str(e)}")


class OpenAIView(APIView):
    authentication_classes = [ChatTokenAuth]

    @extend_schema(
        methods=['POST'],
        description=_('OpenAI Interface Dialogue'),
        summary=_('OpenAI Interface Dialogue'),
        operation_id=_('OpenAI Interface Dialogue'),  # type: ignore
        request=OpenAIAPI.get_request(),
        responses=None,
        tags=[_('Chat')]  # type: ignore
    )
    def post(self, request: Request, application_id: str):
        _log_incoming_request(request)
        return OpenAIChatSerializer(data={'application_id': application_id, 'chat_user_id': request.auth.chat_user_id,
                                          'chat_user_type': request.auth.chat_user_type}).chat(request.data)


class ChatCompletionsView(APIView):
    authentication_classes = [ChatTokenAuth]

    @extend_schema(
        methods=['POST'],
        description=_('OpenAI compatible chat completions'),
        summary=_('Chat completions'),
        operation_id=_('Chat completions'),  # type: ignore
        request=OpenAIAPI.get_request(),
        responses=None,
        tags=[_('Chat')]  # type: ignore
    )
    def post(self, request: Request):
        _log_incoming_request(request)
        return OpenAIChatSerializer(
            data={
                'application_id': request.auth.application_id,
                'chat_user_id': request.auth.chat_user_id,
                'chat_user_type': request.auth.chat_user_type,
            }
        ).chat(request.data)


class CompletionsView(APIView):
    authentication_classes = [ChatTokenAuth]

    @extend_schema(
        methods=['POST'],
        description=_('OpenAI/Ollama compatible text completion'),
        summary=_('Text completion'),
        operation_id=_('Text completion'),  # type: ignore
        request=CompletionsRequestSerializer,
        responses=CompletionsResponseSerializer,
        tags=[_('Chat')]  # type: ignore
    )
    def post(self, request: Request):
        print(request.data, flush=True)
        _log_incoming_request(request)
        serializer = CompletionsRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        model_value = serializer.validated_data['model']
        prompt = serializer.validated_data['prompt']
        system = serializer.validated_data.get('system', '')
        history = serializer.validated_data.get('history', [])
        model_params_setting = serializer.validated_data.get('model_params_setting', {})

        application = QuerySet(Application).filter(id=request.auth.application_id).first()
        if application is None:
            return JsonResponse({'error': {'message': _('Application does not exist')}}, status=404)

        model = None
        try:
            model = QuerySet(Model).filter(workspace_id=application.workspace_id, id=model_value).first()
        except Exception:
            model = None
        if model is None:
            model = QuerySet(Model).filter(workspace_id=application.workspace_id, model_name=model_value).first()
        if model is None:
            model = QuerySet(Model).filter(workspace_id=application.workspace_id, name=model_value).first()
        if model is None:
            return JsonResponse({'error': {'message': _('Model does not exist')}}, status=404)

        chat_model = get_model_instance_by_model_workspace_id(str(model.id), application.workspace_id, **model_params_setting)

        message_list = []
        if system:
            message_list.append(SystemMessage(content=system))
        for m in history:
            role = m.get('role')
            content = m.get('content', '')
            if role == 'system':
                message_list.append(SystemMessage(content=content))
            elif role == 'assistant':
                message_list.append(AIMessage(content=content))
            else:
                message_list.append(HumanMessage(content=content))
        message_list.append(HumanMessage(content=prompt))

        try:
            r = chat_model.invoke(message_list)
        except Exception as e:
            return JsonResponse({'error': {'message': str(e)}}, status=500)

        completion_text = getattr(r, 'content', None)
        if completion_text is None:
            completion_text = str(r)

        try:
            prompt_tokens = chat_model.get_num_tokens_from_messages(message_list)
        except Exception:
            tokenizer = TokenizerManage.get_tokenizer()
            prompt_tokens = sum(len(tokenizer.encode(getattr(m, 'content', '') or '')) for m in message_list)

        try:
            completion_tokens = chat_model.get_num_tokens(completion_text)
        except Exception:
            tokenizer = TokenizerManage.get_tokenizer()
            completion_tokens = len(tokenizer.encode(completion_text))

        response_data = {
            'id': f'cmpl-{str(time.time_ns())}',
            'object': 'text_completion',
            'created': int(time.time()),
            'model': model.model_name,
            'system_fingerprint': model.meta.get('system_fingerprint', 'fp_maxkb') if isinstance(model.meta, dict) else 'fp_maxkb',
            'choices': [
                {
                    'text': completion_text,
                    'index': 0,
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': int(prompt_tokens),
                'completion_tokens': int(completion_tokens),
                'total_tokens': int(prompt_tokens) + int(completion_tokens),
            }
        }
        return JsonResponse(response_data, status=200)


class AnonymousAuthentication(APIView):
    def options(self, request, *args, **kwargs):
        return HttpResponse(
            headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": "true",
                     "Access-Control-Allow-Methods": "POST",
                     "Access-Control-Allow-Headers": "Origin,Content-Type,Cookie,Accept,Token"}, )

    @extend_schema(
        methods=['POST'],
        description=_('Application Anonymous Certification'),
        summary=_('Application Anonymous Certification'),
        operation_id=_('Application Anonymous Certification'),  # type: ignore
        request=ChatAuthenticationAPI.get_request(),
        responses=None,
        tags=[_('Chat')]  # type: ignore
    )
    def post(self, request: Request):
        return result.success(
            AnonymousAuthenticationSerializer(data={'access_token': request.data.get("access_token")}).auth(
                request),
            headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": "true",
                     "Access-Control-Allow-Methods": "POST",
                     "Access-Control-Allow-Headers": "Origin,Content-Type,Cookie,Accept,Token"}
        )


class ApplicationProfile(APIView):
    authentication_classes = [ChatTokenAuth]

    @extend_schema(
        methods=['GET'],
        description=_("Get application related information"),
        summary=_("Get application related information"),
        operation_id=_("Get application related information"),  # type: ignore
        request=None,
        responses=None,
        tags=[_('Chat')]  # type: ignore
    )
    def get(self, request: Request):
        if isinstance(request.auth, ChatAuth):
            return result.success(ApplicationProfileSerializer(
                data={'application_id': request.auth.application_id}).profile())
        raise AppAuthenticationFailed(401, "身份异常")


class AuthProfile(APIView):
    @extend_schema(
        methods=['GET'],
        description=_("Get application authentication information"),
        summary=_("Get application authentication information"),
        operation_id=_("Get application authentication information"),  # type: ignore
        parameters=ChatAuthenticationProfileAPI.get_parameters(),
        responses=None,
        tags=[_('Chat')]  # type: ignore
    )
    def get(self, request: Request):
        return result.success(
            AuthProfileSerializer(data={'access_token': request.query_params.get("access_token")}).profile())


class ChatView(APIView):
    authentication_classes = [ChatTokenAuth]

    @extend_schema(
        methods=['POST'],
        description=_("dialogue"),
        summary=_("dialogue"),
        operation_id=_("dialogue"),  # type: ignore
        request=ChatAPI.get_request(),
        parameters=ChatAPI.get_parameters(),
        responses=None,
        tags=[_('Chat')]  # type: ignore
    )
    def post(self, request: Request, chat_id: str):
        return ChatSerializers(data={'chat_id': chat_id,
                                     'chat_user_id': request.auth.chat_user_id,
                                     'chat_user_type': request.auth.chat_user_type,
                                     'application_id': request.auth.application_id,
                                     'debug': False}
                               ).chat(request.data)


class OpenView(APIView):
    authentication_classes = [ChatTokenAuth]

    @extend_schema(
        methods=['GET'],
        description=_("Get the session id according to the application id"),
        summary=_("Get the session id according to the application id"),
        operation_id=_("Get the session id according to the application id"),  # type: ignore
        parameters=ChatOpenAPI.get_parameters(),
        responses=None,
        tags=[_('Chat')]  # type: ignore
    )
    def get(self, request: Request):
        return result.success(OpenChatSerializers(
            data={'application_id': request.auth.application_id,
                  'chat_user_id': request.auth.chat_user_id, 'chat_user_type': request.auth.chat_user_type,
                  'debug': False}).open())


class CaptchaView(APIView):
    @extend_schema(methods=['GET'],
                   summary=_("Get Chat captcha"),
                   description=_("Get Chat captcha"),
                   operation_id=_("Get Chat captcha"),  # type: ignore
                   tags=[_("Chat")],  # type: ignore
                   responses=CaptchaAPI.get_response())
    def get(self, request: Request):
        username = request.query_params.get('username', None)
        accessToken = request.query_params.get('accessToken', None)
        return result.success(CaptchaSerializer().chat_generate(username, 'chat', accessToken))


class SpeechToText(APIView):
    authentication_classes = [ChatTokenAuth]

    @extend_schema(
        methods=['POST'],
        description=_("speech to text"),
        summary=_("speech to text"),
        operation_id=_("speech to text"),  # type: ignore
        request=SpeechToTextAPI.get_request(),
        responses=SpeechToTextAPI.get_response(),
        tags=[_('Chat')]  # type: ignore
    )
    def post(self, request: Request):
        return result.success(
            SpeechToTextSerializers(
                data={'application_id': request.auth.application_id})
            .speech_to_text({'file': request.FILES.get('file')}))


class TextToSpeech(APIView):
    authentication_classes = [ChatTokenAuth]

    @extend_schema(
        methods=['POST'],
        description=_("text to speech"),
        summary=_("text to speech"),
        operation_id=_("text to speech"),  # type: ignore
        request=TextToSpeechAPI.get_request(),
        responses=TextToSpeechAPI.get_response(),
        tags=[_('Chat')]  # type: ignore
    )
    def post(self, request: Request):
        byte_data = TextToSpeechSerializers(
            data={'application_id': request.auth.application_id}).text_to_speech(request.data)
        return HttpResponse(byte_data, status=200, headers={'Content-Type': 'audio/mp3',
                                                            'Content-Disposition': 'attachment; filename="abc.mp3"'})


class UploadFile(APIView):
    authentication_classes = [ChatTokenAuth]
    parser_classes = [MultiPartParser]

    @extend_schema(
        methods=['POST'],
        description=_("Upload files"),
        summary=_("Upload files"),
        operation_id=_("Upload files"),  # type: ignore
        request=TextToSpeechAPI.get_request(),
        responses=TextToSpeechAPI.get_response(),
        tags=[_('Application')]  # type: ignore
    )
    def post(self, request: Request, chat_id: str):
        files = request.FILES.getlist('file')
        file_ids = []
        meta = {}
        for file in files:
            file_url = FileSerializer(
                data={'file': file, 'meta': meta, 'source_id': chat_id, 'source_type': FileSourceType.CHAT, }).upload()
            file_ids.append({'name': file.name, 'url': file_url, 'file_id': file_url.split('/')[-1]})
        return result.success(file_ids)
