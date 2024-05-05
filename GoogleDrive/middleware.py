from django.utils.deprecation import MiddlewareMixin

class ModuleJavaScriptMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        if request.path.endswith('.mjs'):
            response['Content-Type'] = 'application/javascript'
        return response