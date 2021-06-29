#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, sys, glob
import re
import xml
import xml.dom.minidom


def create_xml_atlas(lfiles, foxml, oid="face"):
    """ take a list of files, create a xml file data_set.xml """

    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, "some_tag", None)
    top_element = doc.documentElement

    for i, fn in enumerate(lfiles):
        e = doc.createElement('subject')
        e.setAttribute('id', "subj{}".format(i))

        v = doc.createElement('visit')
        v.setAttribute('id', "experiment")

        f = doc.createElement('filename')
        f.setAttribute('object_id', oid)

        t = doc.createTextNode(os.path.abspath(fn))

        f.appendChild(t)
        v.appendChild(f)
        e.appendChild(v)

        top_element.appendChild(e)

    with open(foxml, "w") as fo:
        fo.write(doc.toprettyxml())

def create_xml_longitudinal_atlas(lfiles, lsbj, foxml):
    """ take a list of files, create a xml file data_set.xml """

    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, "some_tag", None)
    top_element = doc.documentElement

    for i, fn in enumerate(lfiles):
        e = doc.createElement('subject')
        e.setAttribute('id', "subj{}".format(i))

        v = doc.createElement('visit')
        v.setAttribute('id', "experiment")

        f = doc.createElement('filename')
        f.setAttribute('object_id', "face")

        a = doc.createElement('age')
        x = doc.createTextNode(str(lsbj[i]["age"]))
        a.appendChild(x)

        t = doc.createTextNode(fn)

        f.appendChild(t)
        v.appendChild(f)
        v.appendChild(a)
        e.appendChild(v)

        top_element.appendChild(e)

    with open(foxml, "w") as fo:
        fo.write(doc.toprettyxml())

def create_xml_regression(lfiles, lsbj, foxml):
    """
    take a list of files, create a xml file data_set.xml
    only 'one' subject, each case seen as a visit...
    """

    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, "some_tag", None)
    top_element = doc.documentElement

    e = doc.createElement('subject')
    e.setAttribute('id', 'case')

    for i, fn in enumerate(lfiles):
        v = doc.createElement('visit')
        v.setAttribute('id', "subj{}".format(i))

        f = doc.createElement('filename')
        f.setAttribute('object_id', "face")
        t = doc.createTextNode(fn)
        f.appendChild(t)

        a = doc.createElement('age')
        x = doc.createTextNode(str(lsbj[i]["age"]))
        a.appendChild(x)


        v.appendChild(f)
        v.appendChild(a)
        e.appendChild(v)

    top_element.appendChild(e)

    with open(foxml, "w") as fo:
        fo.write(doc.toprettyxml())

if __name__ == "__main__":
    lfiles = glob.glob("data/mesh_*.vtk")
    foxml = sys.argv[1]
    create_xml_atlas(lfiles, foxml)
