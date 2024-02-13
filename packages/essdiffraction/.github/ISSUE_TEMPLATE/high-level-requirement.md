---
name: High-level requirement
about: Describe a high-level requirement
title: ''
labels: requirement
assignees: ''

---

Goal

Executive summary
Context (Background Knowledge)

Link to the document or short description of the context that needs this requirement
References

DOI or link to the references.

These will be mentioned in the documentation of software or interfaces if needed.

So please add official, accessible and validated documents here.
Environment
Assumptions

Describe the assumptions we may have for this requirement.

Example 1:  This method will be used only a few times in a year.

Example 2: Input data is already calibrated.

Example 3: ...


Use-cases

Who (user or consumer) will use this how with what data (which instrument or group of instruments)? What is the expected output?

Example 1: The wavelength-range stored in the NeXus file (previously configured by the user in NICOS) is used automatically in the reduction workflow.

Example 2: Instrument Scientists will use this Jupyter notebook to investigate the intermediate results, plotting A, B, and C.

Example 3: ...
Preconditions

If applicable, list issues or requirements to be done, fixed or implemented before starting working on this requirement. You can also mention specific Jira ticket# or link to another requirement page. Examples: Sample data is created, metadata field securely defined, discussion with someone [link to the meeting]
Interfaces

One or more of interfaces you want, e.g. python module/script/notebook/application/service/modified workflow/enhancement. Give "None" if there is no user-facing interface, e.g., when this is integrated into an existing workflow.
Input/Output
Input

Describe the type and structure of input e.g. a single nexus file/stream/a list of file-names/metadata ...
Output

Describe the type and structure of output e.g. plot/file/text file/hdf5 file/an integer/triggering another workflow ...
Data Processing Procedure

Describe how the input will be processed to the output. In case you have complicated process, you can add a link or a diagram here. If you want to define multiple 'option's, please specify how the option should be decided. For example, by retrieving a flag from the metadata or manually set by a user. Please mention specific algorithm and a link to the reference if applicable.
Test Cases

Describe what kind of sample data we can use, where they are, when they are available, etc. If we can simulate one of the use-cases mentioned above, please describe how.
Performance metric (non-functional requirements)

If applicable, describe how we measure the performance of the action of use. e.g. Proportional difference between the output from this new method and the output from the earlier method. e.g. How much time it took to process 10,000 files.
Acceptance Criteria

Describe the condition that the development of the requirement is "done".
Potential Effects

If applicable, describe the potential outcome of implementation of the requirement. Examples: It will break [...] type of workflow in [...] instrument experiments. The data reduction on the [...] dataset should be done again. It will make everybody happy.
Followup Work

If applicable, list potential works that need to be followed by this requirement implementation. It can be simply done by adding a link to another requirement page or mentioning Jira tickets or adding issue links Some of them may be the preconditions of another requirement. e.g. Update this and that plots in some documents
