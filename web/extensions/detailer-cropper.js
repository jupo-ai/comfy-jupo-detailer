import { app } from "../../../scripts/app.js";
import { mkName } from "../utils.js";

const PACKAGE_NAME = "Detailer";
const CLASS_NAMES = [
    mkName(PACKAGE_NAME, "DetailerCropper"), 
    mkName(PACKAGE_NAME, "DetailerOutpainter"),
];

const OUTPUT_RESIZE_WIDGETS = [
    "resize_algorithm",
    "output_resolution",
    "output_width",
    "output_height",
    "output_padding",
];

const WIDGET_ORIGINALS = new WeakMap();

function findWidget(node, name) {
    return node.widgets?.find(w => w.name === name);
}

function rememberWidget(widget) {
    if (!widget || WIDGET_ORIGINALS.has(widget)) return;
    WIDGET_ORIGINALS.set(widget, {
        type: widget.type,
        computeSize: widget.computeSize,
    });
}

function setWidgetVisible(node, name, visible) {
    const widget = findWidget(node, name);
    if (!widget) return;

    rememberWidget(widget);
    const original = WIDGET_ORIGINALS.get(widget);

    widget.hidden = !visible;
    widget.disabled = !visible;
    widget.type = visible ? original.type : `jupo_hidden:${name}`;
    widget.computeSize = visible
        ? original.computeSize
        : () => [0, -4];

    widget.linkedWidgets?.forEach(linkedWidget => {
        rememberWidget(linkedWidget);
        const linkedOriginal = WIDGET_ORIGINALS.get(linkedWidget);
        linkedWidget.hidden = !visible;
        linkedWidget.disabled = !visible;
        linkedWidget.type = visible ? linkedOriginal.type : `jupo_hidden:${name}`;
        linkedWidget.computeSize = visible
            ? linkedOriginal.computeSize
            : () => [0, -4];
    });
}

function updateOutputResizeWidgets(node) {
    const outputResize = findWidget(node, "output_resize")?.value;
    const isKeepAspect = outputResize === "keep aspect";
    const isConstant = outputResize === "constant";
    const doesResize = isKeepAspect || isConstant;

    setWidgetVisible(node, "resize_algorithm", doesResize);
    setWidgetVisible(node, "output_resolution", isKeepAspect);
    setWidgetVisible(node, "output_width", isConstant);
    setWidgetVisible(node, "output_height", isConstant);
    setWidgetVisible(node, "output_padding", doesResize);

    const computedSize = node.computeSize?.();
    if (computedSize && node.size) {
        node.size[1] = computedSize[1];
    }
    app.graph?.setDirtyCanvas(true, true);
}

function interceptWidgetValue(widget, onChange) {
    if (!widget || widget._jupoDetailerOutputResizeIntercepted) return;
    widget._jupoDetailerOutputResizeIntercepted = true;

    let widgetValue = widget.value;
    const descriptor =
        Object.getOwnPropertyDescriptor(widget, "value") ||
        Object.getOwnPropertyDescriptor(Object.getPrototypeOf(widget), "value");

    Object.defineProperty(widget, "value", {
        configurable: true,
        enumerable: true,
        get() {
            return descriptor?.get
                ? descriptor.get.call(widget)
                : widgetValue;
        },
        set(newValue) {
            if (descriptor?.set) {
                descriptor.set.call(widget, newValue);
            } else {
                widgetValue = newValue;
            }
            onChange(newValue);
        },
    });
}


const extension = {
    name: mkName(PACKAGE_NAME, "OutputResizeVisibility"), 

    beforeRegisterNodeDef: function(nodeType, nodeData, app) {
        if (!CLASS_NAMES.includes(nodeType.comfyClass)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);

            OUTPUT_RESIZE_WIDGETS.forEach(name => rememberWidget(findWidget(this, name)));
            interceptWidgetValue(findWidget(this, "output_resize"), () => updateOutputResizeWidgets(this));
            updateOutputResizeWidgets(this);

            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = onConfigure?.apply(this, arguments);
            updateOutputResizeWidgets(this);
            return result;
        };
        
    }, 
};

app.registerExtension(extension);
