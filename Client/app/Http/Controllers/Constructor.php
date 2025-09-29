<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Template;

class Constructor extends Controller
{
    public function createTemplate(Request $request)
    {
        $template = new Template;
        $template->name = $request->input('name');
        return $template->save();
    }

    public function getTemplates()
    {
        return Template::with('elementsTemplate')->get();
    }

    public function deleteTemplate(Request $request)
    {
        return Template::destroy($request->input('id'));
    }

    public function createElement(Request $request)
    {
        $template = Template::find($request->input('id'));
        $element = $template->elementsTemplate()->create([
            'data' => json_encode([$request->input('json')]),
        ]);
    }
}
