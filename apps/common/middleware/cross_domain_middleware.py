# coding=utf-8
"""
    @project: MaxKB
    @Author：虎虎
    @file： cross_domain_middleware.py
    @date：2024/5/8 13:36
    @desc:
"""
from django.http import HttpResponse
from django.utils.deprecation import MiddlewareMixin
import json
import logging

from common.cache_data.application_api_key_cache import get_application_api_key
from common.utils.logger import maxkb_logger


class CrossDomainMiddleware(MiddlewareMixin):

    def process_request(self, request):
        try:
            path = request.path
            if path.startswith('/v1/') or path.startswith('/chat/api/v1/'):
                headers = {k: ('Bearer ****' if k.lower() in {'authorization', 'proxy-authorization'} else v)
                           for k, v in request.headers.items()}
                payload = {
                    'path': path,
                    'method': request.method,
                    'remote_addr': request.META.get('HTTP_X_FORWARDED_FOR') or request.META.get('REMOTE_ADDR'),
                    'headers': headers,
                    'query': dict(request.GET.lists()),
                }
                try:
                    payload['body'] = request.body.decode('utf-8', errors='replace')
                except Exception:
                    payload['body'] = str(request.body)
                msg = f"FAY_REQUEST: {json.dumps(payload, ensure_ascii=False)}"
                logging.getLogger('gunicorn.error').warning(msg)
                maxkb_logger.warning(msg)
                print(msg, flush=True)
        except Exception:
            pass
        if request.method == 'OPTIONS':
            return HttpResponse(status=200,
                                headers={
                                    "Access-Control-Allow-Origin": "*",
                                    "Access-Control-Allow-Methods": "GET,POST,DELETE,PUT",
                                    "Access-Control-Allow-Headers": "Origin,X-Requested-With,Content-Type,Accept,Authorization,token"})

    def process_response(self, request, response):
        auth = request.META.get('HTTP_AUTHORIZATION')
        origin = request.META.get('HTTP_ORIGIN')

        try:
            path = request.path
            if path.startswith('/v1/') or path.startswith('/chat/api/v1/'):
                dumped = json.dumps({
                    'path': path,
                    'method': request.method,
                    'status_code': getattr(response, 'status_code', None),
                }, ensure_ascii=False)
                msg = f"FAY_RESPONSE: {dumped}"
                logging.getLogger('gunicorn.error').warning(msg)
                maxkb_logger.warning(msg)
                print(msg, flush=True)
        except Exception:
            pass

        if auth is not None and any([str(auth).startswith(prefix) for prefix in
                                     ['Bearer application-', 'Bearer agent-']]) and origin is not None:
            application_api_key = get_application_api_key(str(auth), True)
            cross_domain_list = application_api_key.get('cross_domain_list', [])
            allow_cross_domain = application_api_key.get('allow_cross_domain', False)
            if allow_cross_domain:
                response['Access-Control-Allow-Methods'] = 'GET,POST,DELETE,PUT'
                response[
                    'Access-Control-Allow-Headers'] = "Origin,X-Requested-With,Content-Type,Accept,Authorization,token"
                if cross_domain_list is None or len(cross_domain_list) == 0:
                    response['Access-Control-Allow-Origin'] = "*"
                elif cross_domain_list.__contains__(origin):
                    response['Access-Control-Allow-Origin'] = origin
        return response
