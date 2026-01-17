# coding=utf-8
"""
    @project: MaxKB
    @Author：虎
    @file： openai_to_response.py
    @date：2024/9/6 16:08
    @desc:
"""
import time
import json

from django.http import JsonResponse
from rest_framework import status

from common.handle.base_to_response import BaseToResponse


class OpenaiToResponse(BaseToResponse):
    def to_block_response(self, chat_id, chat_record_id, content, is_end, prompt_tokens, completion_tokens,
                          other_params: dict = None,
                          _status=status.HTTP_200_OK):
        if other_params is None:
            other_params = {}
        model = other_params.get('model') or 'maxkb'
        data = {
            'id': str(chat_record_id),
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': str(model),
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': content,
                    },
                    'finish_reason': 'stop',
                }
            ],
            'usage': {
                'prompt_tokens': int(prompt_tokens or 0),
                'completion_tokens': int(completion_tokens or 0),
                'total_tokens': int((prompt_tokens or 0) + (completion_tokens or 0)),
            },
        }
        return JsonResponse(data=data, status=_status, json_dumps_params={'ensure_ascii': False})

    def to_stream_chunk_response(self, chat_id, chat_record_id, node_id, up_node_id_list, content, is_end,
                                 prompt_tokens,
                                 completion_tokens, other_params: dict = None):
        if other_params is None:
            other_params = {}
        model = other_params.get('model') or 'maxkb'
        delta = {}
        if content:
            delta['content'] = content
        chunk = {
            'id': str(chat_record_id),
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': str(model),
            'choices': [
                {
                    'index': 0,
                    'delta': delta,
                    'finish_reason': 'stop' if is_end else None,
                }
            ],
        }
        chunk_str = json.dumps(chunk, ensure_ascii=False)
        if is_end:
            return super().format_stream_chunk(chunk_str) + 'data: [DONE]\n\n'
        return super().format_stream_chunk(chunk_str)
