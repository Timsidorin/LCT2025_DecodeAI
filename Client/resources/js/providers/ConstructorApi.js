import {BaseApi} from "./BaseApi.js";

export class ConstructorApi extends BaseApi{
    constructor() {
        super(__BASE__LARAVEL__URL__);
    }

    async createTemplate(data) {
        try {
            super.httpMethod = 'post';
            super.sourceUrl = '/api/template';
            super.data = data;
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }

    async getTemplates() {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/template';
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }

    async deleteTemplate(id) {
        try {
            super.httpMethod = 'delete';
            super.sourceUrl = '/api/template';
            super.data = {id}
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }

    async createElement(data) {
        try {
            super.httpMethod = 'post';
            super.sourceUrl = '/api/element';
            super.data = data;
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }
}
