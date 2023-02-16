var dragTimer;

function addInputFileDragListener(form)
{
    var element = form.querySelector(".input-file-upload");

    var borderDrag = '3px dashed crimson';
    var bgColorDrag = "#ffefef";
    var borderOld = element.style.border;
    var bgColorOld = element.style.backgroundColor;


    $(document).on('dragover', function(e) {
        var dt = e.originalEvent.dataTransfer;
        if (dt.types && (dt.types.indexOf ? dt.types.indexOf('Files') != -1 : dt.types.contains('Files'))) {
          element.style.border = borderDrag;
          element.style.backgroundColor = bgColorDrag;
          window.clearTimeout(dragTimer);
        }
    });
    $(document).on('dragleave', function(e) {
        dragTimer = window.setTimeout(function() {
            element.style.border = borderOld;
            element.style.backgroundColor = bgColorOld;
        }, 25);
    });
    
      

}