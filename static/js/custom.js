function checkMaskImage()
{
    var maskImage = document.getElementById("maskImage").checked;
    var element = document.getElementById("maskImageDisplay");
    if (maskImage)
        element.classList.remove("d-none");
    else
    element.classList.add("d-none");
}